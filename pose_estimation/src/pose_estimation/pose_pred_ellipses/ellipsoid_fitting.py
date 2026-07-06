import numpy as np
from scipy.spatial import ConvexHull, QhullError
import open3d as o3d
import logging
from abc import ABC, abstractmethod
import torch
from torch.optim import Adam
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns
from typing import Literal


from shared.se3_utilities import r_t_to_hom
from shared.assertion_helpers import assert_homogeneous_mat

from .ellipsoid_utilities_numpy import create_ellipsoid_lineset, assert_primal_quadratic_hom_ellipsoid

from ..utilities.pose_optimisation import AdamConfig, compute_pose_exp_se3, compute_pose_euler
from ..utilities.point_utilities import remove_outliers_from_point_cloud



def abc_and_base_t_ellipsoid_2_primal_quadratic(abc:np.ndarray, base_t_ellipsoid:np.ndarray)->np.ndarray:
    assert abc.ndim == 1 and abc.shape[0] == 3, f"abc has wrong shape: {abc}"
    assert assert_homogeneous_mat(base_t_ellipsoid, size=4)

    a, b, c = abc
    q_ellipsoid = np.diag([1/a**2, 1/b**2, 1/c**2, -1.0])

    ellipsoid_t_base = np.linalg.inv(base_t_ellipsoid)
    primal_quadratic = ellipsoid_t_base.T @ q_ellipsoid @ ellipsoid_t_base

    return primal_quadratic


class EllipsoidFitter(ABC):
    def __init__(self, min_num_points:int = 10, contamination:float = 0.05, visualize:bool = False):
        self.min_num_points = min_num_points
        self.contamination = contamination
        self.visualize = visualize

    @abstractmethod
    def fit_ellipsoid(self, points:np.ndarray)->tuple[np.ndarray, np.ndarray] | None:
        """
        :param point_cloud: A Nx3 point cloud in the base_frame
        :param config: How to do it
        :return: a tuple of the base_t_ellipsoid hom. mtx (4x4) and the primal quadratics (4x4) or None if fitting failed
        """
        pass

    def prep_point_cloud(self, points:np.ndarray)->np.ndarray | None:
        assert points.ndim == 2 and points.shape[-1] == 3, f"Not valid point format: {points.shape}"

        points = points[np.isfinite(points).all(axis=-1)]

        if points.shape[0] < self.min_num_points:
            return None

        points = remove_outliers_from_point_cloud(points, contamination=self.contamination)

        if points.shape[0] < self.min_num_points:
            return None
        
        return points


    def calculate_initial_params(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        pc_red = self.prep_point_cloud(points=points)

        if pc_red is None:
            return None
        else:
            point_cloud = pc_red

        center = point_cloud.mean(axis=0)
        pts_centered = point_cloud - center

        cov_matrix = np.cov(pts_centered, rowvar=False)

        eigvals, eigvecs = np.linalg.eigh(cov_matrix)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]

        if np.linalg.det(eigvecs) < 0:
            eigvecs[:, -1] *= -1

        base_t_ellipsoid = r_t_to_hom(eigvecs, center)
        pts_local = (base_t_ellipsoid[:3, :3].T @ pts_centered.T).T

        a = np.max(np.abs(pts_local[:, 0]))
        b = np.max(np.abs(pts_local[:, 1]))
        c = np.max(np.abs(pts_local[:, 2]))

        return np.array([a, b, c]), base_t_ellipsoid, point_cloud


    @staticmethod
    def visualize_ellipsoid_fit(point_cloud:np.ndarray, base_t_ellipsoid:np.ndarray, primal_quadratic:np.ndarray):
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        to_vis = [base_frame]
        ls = create_ellipsoid_lineset(base_t_ellipsoid, primal_quadratic)

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(point_cloud)
        pcd1.paint_uniform_color([1,0,0])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(base_t_ellipsoid)

        to_vis += [base_frame, ls, pcd1, frame]
        o3d.visualization.draw_geometries(to_vis, f"Ellipsoid fit visualization")


class SimpleEllipsoidFitter(EllipsoidFitter):
    def __init__(self, min_num_points: int = 10, contamination: float = 0.05, visualize: bool = False):
        super().__init__(
            min_num_points=min_num_points,
            contamination=contamination,
            visualize=visualize
        )

    def fit_ellipsoid(self, points: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:

        abc__base_t_ellipsoid = self.calculate_initial_params(points)
        if abc__base_t_ellipsoid is None:
            return None

        abc, base_t_ellipsoid, _ = abc__base_t_ellipsoid

        primal_quadratic = abc_and_base_t_ellipsoid_2_primal_quadratic(abc=abc, base_t_ellipsoid=base_t_ellipsoid)

        assert assert_primal_quadratic_hom_ellipsoid(primal_quadratic)

        if self.visualize:
            self.visualize_ellipsoid_fit(
                point_cloud=points,
                base_t_ellipsoid=base_t_ellipsoid,
                primal_quadratic=primal_quadratic
            )

        return base_t_ellipsoid, primal_quadratic


class LeastShellDistanceEllipsoidFitter(EllipsoidFitter):
    def __init__(
            self, 
            min_num_points:int = 10, 
            contamination:float = 0.05, 
            visualize:bool = False, 
            use_cnvx_hull:bool = True,
            adam_config:AdamConfig = AdamConfig(learning_rate=0.001, max_itterations=1000),
            size_penalty:float = 0.95,
            delta_pose_mapping:Literal["euler", "se3_exp"] = "se3_exp",
            distance_p_norm: Literal['-inf', 'inf'] | int = 1,
            size_p_norm: Literal['-inf', 'inf'] | int = 1,
            device:Literal['cuda', 'cpu'] = 'cpu',
            gather_losses:bool = False
        ):
        super().__init__(
            min_num_points=min_num_points,
            contamination=contamination, 
            visualize=visualize,
        )
        self.use_cnvx_hull = use_cnvx_hull
        self.adam_config = adam_config
        self.size_penalty = size_penalty
        self.distance_p_norm = distance_p_norm
        self.size_p_norm = size_p_norm
        
        if delta_pose_mapping == "se3_exp":
            self.apply_delta_pose = compute_pose_exp_se3
        else:
            self.apply_delta_pose = compute_pose_euler
        
        self.device = torch.device(device)

        self.accumulated_losses = [] if gather_losses else None



    def visualize_optimisation_losses(self, ax:Axes):
        if self.accumulated_losses is None:
            return
        
        rows = []
        for run_idx, loss_list in enumerate(self.accumulated_losses):
            for iteration, loss in enumerate(loss_list):
                rows.append({"run": run_idx, "iteration": iteration, "loss": loss,})

        df = pd.DataFrame(rows)
        sns.lineplot(data=df, ax = ax, x="iteration", y="loss",hue="run")


    @staticmethod
    def compute_algebraic_distance(
            points: torch.Tensor,
            base_t_ellipsoid: torch.Tensor,
            abc: torch.Tensor
    ) -> torch.Tensor:
        R = base_t_ellipsoid[:3, :3]
        t = base_t_ellipsoid[:3, 3]

        # Transform to local coordinates: p_local = R^T * (p_global - t)
        pts_local = (R.T @ (points - t.unsqueeze(0)).T).T

        # Compute algebraic error: F = x^2/a^2 + y^2/b^2 + z^2/c^2 - 1
        abc_clamped = torch.clamp(abc, min=1e-8)
        error = torch.sum((pts_local / abc_clamped.unsqueeze(0)) ** 2, dim=1) - 1.0

        return error


    def fit_ellipsoid(self, points:np.ndarray)->tuple[np.ndarray, np.ndarray] | None:
        
        abc__base_t_ellipsoid__points = self.calculate_initial_params(points)
        if abc__base_t_ellipsoid__points is None:
            return None
        
        abc_init, base_t_ellipsoid_init, points_np = abc__base_t_ellipsoid__points

        if self.use_cnvx_hull:
            hull = ConvexHull(points_np)
            points_np = points_np[hull.vertices]


        abc_init_torch = torch.tensor(abc_init, device=self.device, dtype=torch.float32)
        base_t_ellipsoid_init_torch = torch.tensor(base_t_ellipsoid_init, dtype=torch.float32)
        points_torch = torch.tensor(points_np, device=self.device, dtype=torch.float32)

        params = torch.nn.Parameter(torch.zeros(9, dtype=torch.float32, device=self.device))
        params.data[6:9] = abc_init_torch
        

        def get_loss():
            base_t_ellipsoid = self.apply_delta_pose(x_i = params[:6], cam_t_base=base_t_ellipsoid_init_torch)
            distances = LeastShellDistanceEllipsoidFitter.compute_algebraic_distance(
                points=points_torch,
                base_t_ellipsoid=base_t_ellipsoid,
                abc=params[6:]
            )
            return (
                (1-self.size_penalty)* torch.linalg.norm(distances, ord = self.distance_p_norm) 
                + self.size_penalty * torch.linalg.norm(params[6:], ord = self.size_p_norm)
            )

        optimizer = Adam(
            [params],
            lr=self.adam_config.learning_rate,
            betas=self.adam_config.betas,
            eps=self.adam_config.eps
        )

        losses = []
        for _ in range(self.adam_config.max_itterations):
            optimizer.zero_grad()
            loss = get_loss()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        if self.accumulated_losses is not None:
            self.accumulated_losses.append(losses)

        final_abc = params[6:9].detach().cpu().numpy()

        final_base_t_ellipsoid = self.apply_delta_pose(x_i = params[:6], cam_t_base=base_t_ellipsoid_init_torch).detach().cpu().numpy()

        primal_quadratic = abc_and_base_t_ellipsoid_2_primal_quadratic(abc=final_abc, base_t_ellipsoid=final_base_t_ellipsoid)

        assert assert_primal_quadratic_hom_ellipsoid(primal_quadratic)

        if self.visualize:
            self.visualize_ellipsoid_fit(
                point_cloud=points_np,
                base_t_ellipsoid = final_base_t_ellipsoid,
                primal_quadratic=primal_quadratic
            )
        
        return final_base_t_ellipsoid, primal_quadratic


    
class MVEEEllipsoidFitter(EllipsoidFitter):
    def __init__(
            self, 
            min_num_points:int = 4, 
            contamination:float = 0.05, 
            visualize:bool = False,
            tolerance: float = 1e-3,
            max_iterations: int = 20000,
            use_convex_hull: bool = True,
        ):
        """
        A Minimal volume enclosing ellipsoid fitter.
        :param min_num_points: Minimal number of points to fit an ellipsoid
        :param contamination: The contamination for i-Forest pointcloud cleanup
        :param visualize: if true will visualize each fit
        :param max_itterations: The max number of itterations used by the fitter
        :param tolerance: Convergence threshhold, smaller -> more accurate but more itterations
        :param use_convex_hull: If true will fit to the convex hull for fitting
        """

        super().__init__(
            min_num_points=min_num_points,
            contamination=contamination, 
            visualize=visualize
        )

        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.use_convex_hull = use_convex_hull


    def fit_ellipsoid(self, points: np.ndarray)->tuple[np.ndarray, np.ndarray] | None:

        points_clean = self.prep_point_cloud(points=points)

        if points_clean is None:
            return None

        fitting_points = points_clean
        if self.use_convex_hull and len(points_clean) > 20:
            try:
                hull = ConvexHull(points_clean)
                fitting_points = points_clean[hull.vertices]
            except QhullError:
                logging.debug("Convex hull generation failed")

        if np.linalg.matrix_rank(fitting_points - np.mean(fitting_points, axis=0)) < 3:
            logging.debug(
                "The points are coplanar or nearly coplanar. "
                "A finite 3D enclosing ellipsoid cannot be determined."
            )
            return None

        offset = np.mean(fitting_points, axis=0)
        scale = np.std(fitting_points, axis=0)

        cloud_size = np.linalg.norm(np.ptp(fitting_points, axis=0))
        scale = np.maximum(scale, (cloud_size * 1e-12).clip(min = 1e-12))
        normalized_points = (fitting_points - offset) / scale

        number_of_points, _ = normalized_points.shape

        Q = np.vstack([normalized_points.T,np.ones(number_of_points)]) # [4, m]

        weights = np.full(number_of_points, 1.0 / number_of_points,)

        # 2. Khachiyan MVEE 
        for _ in range(self.max_iterations):
            X = (Q * weights) @ Q.T

            X_inverse = np.linalg.pinv(X,rcond=1e-14)

            M = np.einsum("ij,ji->i",Q.T @ X_inverse,Q, optimize=True)

            maximum_index = int(np.argmax(M))
            maximum_M = float(M[maximum_index])

            convergence_threshold = (1.0 + self.tolerance) * 4.0

            if maximum_M <= convergence_threshold:
                break

            step_size = (maximum_M - 4.0) / (4.0 * (maximum_M - 1.0))

            step_size = float(np.clip(step_size, 0.0, 1.0))

            weights *= 1.0 - step_size
            weights[maximum_index] += step_size


        normalized_center = normalized_points.T @ weights

        covariance_like = (normalized_points.T * weights) @ normalized_points - np.outer(normalized_center, normalized_center,)

        normalized_A = np.linalg.pinv(covariance_like,rcond=1e-14) / 3

        # Change back to original scale and offset
        center = offset + scale * normalized_center
        inverse_scale = np.diag(1.0 / scale)
        A = inverse_scale @ normalized_A @ inverse_scale
        A = 0.5 * (A + A.T)

        # ---------------------------------------------------------
        # 4. A Rotation from the eigenvectors of A, axes lengths from the eigenvalues
        # ---------------------------------------------------------
        eigenvalues, eigenvectors = np.linalg.eigh(A)

        if np.any(eigenvalues <= 0):
            logging.debug("Failed to obtain a positive-definite ellipsoid matrix")
            return None

        # smallest eigenvalue corresponds to largest axis, sort in ascending order
        order = np.argsort(eigenvalues)

        eigenvalues = eigenvalues[order]
        rotation = eigenvectors[:, order]

        axes = 1.0 / np.sqrt(eigenvalues)

        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1

        base_t_ellipsoid = r_t_to_hom(rotation, center)

        primal_quadratic = abc_and_base_t_ellipsoid_2_primal_quadratic(abc=axes, base_t_ellipsoid=base_t_ellipsoid)

        if self.visualize:
            self.visualize_ellipsoid_fit(
                point_cloud=points,
                base_t_ellipsoid = base_t_ellipsoid,
                primal_quadratic=primal_quadratic
            )

        return base_t_ellipsoid, primal_quadratic