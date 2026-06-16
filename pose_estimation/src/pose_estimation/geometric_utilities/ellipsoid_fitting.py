import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.linalg import eig
from dataclasses import dataclass
import open3d as o3d
import logging
from abc import ABC, abstractmethod

from shared.se3_utilities import r_t_to_hom

from .point_utilities import remove_outliers_from_point_cloud
from .ellipsoid_utilities_numpy import sample_points_in_primal_quadratic, assert_primal_quadratic_hom_ellipsoid



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
        assert points.ndim == 2 and points.shape[-1] == 3

        points = points[np.isfinite(points).all(axis=-1)]

        if points.shape[0] < self.min_num_points:
            return None

        points = remove_outliers_from_point_cloud(points, contamination=self.contamination)

        if points.shape[0] < self.min_num_points:
            return None
        
        return points

    @staticmethod
    def visualize_ellipsoid_fit(point_cloud:np.ndarray, base_t_ellipsoid:np.ndarray, primal_quadratic:np.ndarray):
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        to_vis = [base_frame]
        pc_np = sample_points_in_primal_quadratic(base_t_ellipsoid, primal_quadratic)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)
        pcd.paint_uniform_color([0,1,0])

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(point_cloud)
        pcd1.paint_uniform_color([1,0,0])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(base_t_ellipsoid)

        to_vis += [pcd, pcd1, frame]
        o3d.visualization.draw_geometries(to_vis, f"Ellipsoid fit visualization")   


class SimpleEllipsoidFitter(EllipsoidFitter):
    def __init__(self, min_num_points:int = 10, contamination:float = 0.05, visualize:bool = False):
        super().__init__(
            min_num_points=min_num_points,
            contamination=contamination, 
            visualize=visualize
        )

    def fit_ellipsoid(self, points:np.ndarray)->tuple[np.ndarray, np.ndarray] | None:

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

        q_ellipsoid = np.diag([1/a**2, 1/b**2, 1/c**2, -1.0])

        ellipsoid_t_base = np.linalg.inv(base_t_ellipsoid)
        primal_quadratic = ellipsoid_t_base.T @ q_ellipsoid @ ellipsoid_t_base

        assert assert_primal_quadratic_hom_ellipsoid(primal_quadratic)

        if self.visualize:
            self.visualize_ellipsoid_fit(
                point_cloud=point_cloud,
                base_t_ellipsoid = base_t_ellipsoid,
                primal_quadratic=primal_quadratic
            )
        
        return base_t_ellipsoid, primal_quadratic
    
class MVEEEllipsoidFitter(EllipsoidFitter):
    def __init__(
            self, 
            min_num_points:int = 4, 
            contamination:float = 0.05, 
            visualize:bool = False,
            tolerance: float = 1e-3,
            max_iterations: int = 20000,
            margin: float = 1e-6,
            use_convex_hull: bool = True,
        ):
        """
        MVEE。

        Defination:
            sum(((p - center) @ rotation / axes) ** 2) <= 1

        Parameters
        ----------
        points : ndarray, shape (N, 3)
            Input 3D points to be enclosed by the ellipsoid.

        tolerance : float
            Khachiyan threshold for convergence. Smaller values lead to a more accurate ellipsoid but require more iterations.
        max_iterations : int
        Maximum number of iterations for the Khachiyan algorithm.

        margin : float
            scale the axes by (1 + margin) to ensure all points are enclosed, accounting for numerical issues.

        use_convex_hull : bool
            If True, only use the convex hull vertices of the input points for fitting to speed up the algorithm.
        """

        super().__init__(
            min_num_points=min_num_points,
            contamination=contamination, 
            visualize=visualize
        )

        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.margin = margin
        self.use_convex_hull = use_convex_hull


    def fit_ellipsoid(self, points: np.ndarray)->tuple[np.ndarray, np.ndarray] | None:

        points_clean = self.prep_point_cloud(points=points)

        if points_clean is None:
            return None
        else:
            points = points_clean

        # all input points will be checked for enclosure at the end, 
        all_points = points

        # MVEE only depends on the convex hull vertices, so we can optionally filter the input points to speed up the algorithm. However, we still need to check all input points at the end to ensure they are enclosed.
        fitting_points = all_points

        if self.use_convex_hull and len(all_points) > 20:
            try:
                hull = ConvexHull(all_points)
                fitting_points = all_points[hull.vertices]
            except QhullError:
                fitting_points = all_points

        if np.linalg.matrix_rank(
            fitting_points - np.mean(fitting_points, axis=0)
        ) < 3:
            logging.debug(
                "The points are coplanar or nearly coplanar. "
                "A finite 3D enclosing ellipsoid cannot be determined."
            )
            return None

        # ---------------------------------------------------------
        # 1. Nomilization and setup for Khachiyan MVEE algorithm
        # ---------------------------------------------------------
        offset = np.mean(fitting_points, axis=0)
        scale = np.std(fitting_points, axis=0)

        cloud_size = np.linalg.norm(
            np.ptp(fitting_points, axis=0)
        )

        minimum_scale = max(
            cloud_size * 1e-12,
            1e-12,
        )

        scale = np.maximum(scale, minimum_scale)

        normalized_points = (
            fitting_points - offset
        ) / scale

        number_of_points, dimension = normalized_points.shape

        # Q shape: (dimension + 1, number_of_points)
        Q = np.vstack([
            normalized_points.T,
            np.ones(number_of_points),
        ])

        weights = np.full(
            number_of_points,
            1.0 / number_of_points,
        )

        maximum_M = np.inf

        # ---------------------------------------------------------
        # 2. Khachiyan MVEE 
        # ---------------------------------------------------------
        for iteration in range(self.max_iterations):
            X = (Q * weights) @ Q.T

            X_inverse = np.linalg.pinv(X,rcond=1e-14)

            M = np.einsum("ij,ji->i",Q.T @ X_inverse,Q, optimize=True)

            maximum_index = int(np.argmax(M))
            maximum_M = float(M[maximum_index])

            convergence_threshold = (1.0 + self.tolerance) * (dimension + 1)

            if maximum_M <= convergence_threshold:
                break

            step_size = (maximum_M - dimension - 1.0) / ((dimension + 1.0) * (maximum_M - 1.0))

            step_size = float(np.clip(step_size, 0.0, 1.0))

            weights *= 1.0 - step_size
            weights[maximum_index] += step_size

        # ---------------------------------------------------------
        # 3. In the normalized space, the MVEE matrix A can be directly computed from the optimal weights.
        # ---------------------------------------------------------
        normalized_center = normalized_points.T @ weights

        covariance_like = (
            (normalized_points.T * weights)
            @ normalized_points
            - np.outer(
                normalized_center,
                normalized_center,
            )
        )

        normalized_A = np.linalg.pinv(covariance_like,rcond=1e-14,) / dimension

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

        # ---------------------------------------------------------
        # 5. Check all input points are enclosed, and apply margin inflation
        # ---------------------------------------------------------
        local_points = (
            all_points - center
        ) @ rotation

        ellipsoid_values = np.sum(
            (local_points / axes) ** 2,
            axis=1,
        )

        maximum_value_before_inflation = float(np.max(ellipsoid_values))

        # If the maximum value is slightly above 1 due to numerical issues, we can inflate the axes to ensure all points are enclosed.
        inflation_factor = np.sqrt(max(1.0, maximum_value_before_inflation))

        inflation_factor *= 1.0 + self.margin
        axes *= inflation_factor

        base_t_ellipsoid = r_t_to_hom(rotation, center)

        q_ellipsoid = np.diag([
            1.0 / axes[0]**2,
            1.0 / axes[1]**2,
            1.0 / axes[2]**2,
            -1.0
        ])

        ellipsoid_t_base = np.linalg.inv(base_t_ellipsoid)

        primal_quadratic = ellipsoid_t_base.T @ q_ellipsoid @ ellipsoid_t_base

        if self.visualize:
            self.visualize_ellipsoid_fit(
                point_cloud=points,
                base_t_ellipsoid = base_t_ellipsoid,
                primal_quadratic=primal_quadratic
            )

        return base_t_ellipsoid, primal_quadratic