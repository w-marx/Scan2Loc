import numpy as np
from scipy.spatial.transform import Rotation
import torch
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib.axes import Axes
from abc import ABC, abstractmethod
import seaborn as sns
import pandas as pd

from .ellipsoid_utilities_numpy import gaussian_ellipse_s_to_matplotlib_ellipse_s


def hom_to_quat_vec(hom_mat:np.ndarray)->np.ndarray:
    """
    Convert 4x4 homogenous matrix to 7 element quaternion + translation vector
    """
    assert hom_mat.shape == (3,4) or hom_mat.shape == (4,4), f"wrong shape {hom_mat.shape}"
    R = hom_mat[:3, :3]
    t = hom_mat[:3, 3]

    quat = Rotation.from_matrix(R).as_quat()
    quat_vec = np.array([t[0], t[1], t[2], quat[0], quat[1], quat[2], quat[3]])
    return quat_vec


def project_dual_quadratics_to_primal_conicals_torch(
    dual_quadratics:torch.Tensor,
    cam_t_base:torch.Tensor,
    intrinsic_mtx:torch.Tensor,
)-> torch.Tensor:
    """
    Takes Nx4x4 dual quadratics in the world frame
    and returns the Nx3x3 primal conics in the camera frame
    :param dual_quadratics: The duals quadratic as a Nx4x4 matrix
    :param cam_t_base: A 4x4 camera->base homogeneous transformation matrix
    :param intrinsic_mtx: The 3x3 intrinsic matrix of the camera
    :return: The 3x3 primal conic of the projected ellipse
    """
    P = cam_t_base[:3, :] # [N, 3, 4]
    KP = intrinsic_mtx @ P # [N, 3, 4]
    cam_dual_conic = KP.unsqueeze(0) @ dual_quadratics @ KP.mT.unsqueeze(0) # [N, 3, 3]
    cam_primal_conic = torch.linalg.inv(cam_dual_conic) # [N, 3 , 3]

    return (cam_primal_conic+cam_primal_conic.transpose(-2,-1))/2 # [N, 3, 3]


def primal_conics_to_gaussian_ellipses_torch(
        primal_conic_s:torch.Tensor,
        )-> tuple[torch.Tensor, torch.Tensor]:
    """
    Takes Nx3x3 primal conics and returns normal distributions where the edge of the elipsoid is the Mahalanobis of one
    :param primal_conic_s: The Nx3x3 primal conics
    :return a tuple: Nx2 of the mu's and Nx2x2 of the sigma's
    """
    a = primal_conic_s[:, 0:2, 0:2] # [N, 2, 2]
    b = primal_conic_s[:, 0:2, 2] # [N, 2]
    c = primal_conic_s[:, 2, 2] # [N]
    mu = -torch.linalg.solve(a, b) 

    # normalization
    s = (mu.unsqueeze(1) @ a @ mu.unsqueeze(-1)).squeeze() - c
    s = torch.clamp(s, min = 1e-8) # [N]
    a_norm = primal_conic_s[:, :2, :2] / s[:, None, None] # [N, 2, 2]

    # Force symmetry
    a_norm = 0.5 * (a_norm + a_norm.transpose(-2, -1)) # [N, 2, 2]
    sigma = torch.linalg.inv(a_norm) # [N, 2, 2]

    return mu, sigma


def wasserstein_distances_sq_torch(
        mu1_s:torch.Tensor,
        sigma1_s:torch.Tensor,
        mu2_s:torch.Tensor,
        sigma2_s:torch.Tensor,
        sigma2_s_sqrt:torch.Tensor
    ):
    """
    Computes the squared wasserstein distances
    :param mu1_s: An Nx2 array of means
    :param sigma1_s: An Nx2x2 array of covariance matrices
    :param mu2_s: An Nx2 array of means
    :param sigma2_s: An Nx2x2 array of covariance matrices
    :param sigma2_s_sqrt: The roots of the covariance matrices
    :returns an array of length N of the distances
    """

    mean_dist_sq = torch.sum((mu1_s-mu2_s)**2, axis = 1)
    cross_sq = sigma2_s_sqrt @ sigma1_s @ sigma2_s_sqrt
    cross = sqrtm_2x2_torch(cross_sq)
    dist_sq = mean_dist_sq + (sigma1_s+sigma2_s-2*cross).diagonal(dim1 = -2, dim2 = -1).sum(-1)
    return dist_sq


def sqrtm_2x2_torch(matrices:torch.Tensor):
    """
    Computes the Square root of an 2x2 matrix
    :param matrices: Nx2x2 matrix array
    :return Nx2x2 array with sqrt(m)
    """
    vals, vecs = torch.linalg.eigh(matrices)
    sqrt_vals = torch.sqrt(vals.clamp(min = 1e-9))
    return (vecs * sqrt_vals.unsqueeze(-2)) @ vecs.mT


def full_error_calculation(
        dual_quadratics:torch.Tensor,
        cam_t_base:torch.Tensor,
        intrinsic_mtx:torch.Tensor,
        obs_gaussians_mu_s:torch.Tensor,
        obs_gaussians_sigma_s:torch.Tensor,
        obs_sigma_s_sqrt:torch.Tensor
        )->torch.Tensor:
        proj_primal_conincals = project_dual_quadratics_to_primal_conicals_torch(
            dual_quadratics=dual_quadratics,
            cam_t_base=cam_t_base,
            intrinsic_mtx=intrinsic_mtx
        )
        proj_mu_s, proj_sigma_s = primal_conics_to_gaussian_ellipses_torch(proj_primal_conincals)
        return wasserstein_distances_sq_torch(
            mu1_s=proj_mu_s,
            sigma1_s=proj_sigma_s,
            mu2_s = obs_gaussians_mu_s,
            sigma2_s=obs_gaussians_sigma_s,
            sigma2_s_sqrt=obs_sigma_s_sqrt
        )

full_error_calculation_compiled = torch.compile(full_error_calculation, mode="reduce-overhead")


class PnEOptimizer(ABC):
    def __init__(self, accumulate_losses:bool = False) -> None:
        """
       An PnEOptimizer is an class that optimized a given pose, so that the difference between projected and
       observed ellipsoids is minimized.
        """
        super().__init__()
        self.fig, self.axes = None, None
        self._vis_img_rgb = None
        self._vis_intrinsic_mtx = None
        self._vis_dual_quadratics = None
        self.accumulated_losses = [] if accumulate_losses else None


    def visualize_opt_losses(self, ax:Axes):
        if self.accumulated_losses is None:
            return

        rows = []
        for run_idx, loss_list in enumerate(self.accumulated_losses):
            for iteration, loss in enumerate(loss_list):
                rows.append({"run": run_idx, "iteration": iteration, "loss": loss,})
        df = pd.DataFrame(rows)
        sns.lineplot(data=df, ax = ax, x="iteration", y="loss",hue="run")


    @abstractmethod
    def optimize_pne(
        self,
        initial_cam_t_base:np.ndarray,
        primal_quadratics:np.ndarray,
        primal_conicals:np.ndarray,
        intrinsic_cam_mat:np.ndarray,
        visualize_result:None | np.ndarray = None,
    )->np.ndarray:
        """
        :param initial_cam_t_base: An homogeneous 4x4 matrix of the initial camera pose
        :param primal_quadratics: An Bx4x4 batch of the primal quadratics in world space
        :param primal_conicals: An Bx3x3 batch of the observed primal conicals in camera space
        :param intrinsic_cam_mat: The 3x3 intrinsic camera matrix
        :param visualize_result: either an RGB image as an HxWx3 numpy array or None if visualisation is not wanted
        :return: The optimised 4x4 hom. matrix
        """
        raise NotImplementedError("Optimize PnE not implemented in base class")



    def register_visualisation1(self,
                                img_rgb:np.ndarray, 
                                dual_quadratics:torch.Tensor,
                                cam_t_base:torch.Tensor,
                                intrinsic_mtx:torch.Tensor,
                                obs_gaussians_sigma_s:torch.Tensor,
                                obs_gaussians_mu_s:torch.Tensor,
                                title:str = "Before optimisation"
                                ):
        """
        Draws the given state onto the first of 3 axes
        """
        self.fig, self.axes = plt.subplots(1, 3, figsize=(24, 8))
        self._vis_img_rgb = img_rgb
        self._vis_intrinsic_mtx = intrinsic_mtx
        self._vis_dual_quadratics = dual_quadratics
        self._vis_obs_obs_gaussians_sigma_s = obs_gaussians_sigma_s
        self._vis_obs_gaussians_mu_s = obs_gaussians_mu_s
        self.visualize_errors(
            img_rgb=img_rgb, 
            ax=self.axes[0], 
            dual_quadratics=dual_quadratics,
            cam_t_base=cam_t_base,
            obs_gaussians_mu_s=obs_gaussians_mu_s, 
            obs_gaussians_sigma_s=obs_gaussians_sigma_s,
            title=title, 
            intrinsic_mtx=intrinsic_mtx
        )


    def register_visualisation2(self,cam_t_base:torch.Tensor,title:str = "After optimisation"):
        """
        Draws the state given in `register_visualisation1` from the new perspective onto the second of 3 axes
        """
        if self._vis_img_rgb is not None and self.axes is not None and self._vis_dual_quadratics is not None and self._vis_intrinsic_mtx is not None:
            self.visualize_errors(
                img_rgb=self._vis_img_rgb, 
                ax=self.axes[1], 
                dual_quadratics=self._vis_dual_quadratics,
                cam_t_base=cam_t_base,
                obs_gaussians_mu_s=self._vis_obs_gaussians_mu_s, 
                obs_gaussians_sigma_s=self._vis_obs_obs_gaussians_sigma_s,
                title=title, 
                intrinsic_mtx=self._vis_intrinsic_mtx
            )


    def register_visualisation_loss(self, losses:list[float]):
        """
        Draws a list of losses onto the 3rd of 3 axes
        """
        if self.axes is not None:
            self.axes[2].plot(losses)
            self.axes[2].set_title("Loss over iterations")
            self.axes[2].set_xlabel("Iteration")
            self.axes[2].set_ylabel("Loss")


    @staticmethod    
    def visualize_errors( 
            img_rgb:np.ndarray, 
            ax:Axes,
            dual_quadratics:torch.Tensor,
            cam_t_base:torch.Tensor,
            intrinsic_mtx:torch.Tensor,
            obs_gaussians_sigma_s:torch.Tensor,
            obs_gaussians_mu_s:torch.Tensor,
            title:str = "PnE errors"
        ):
        """
        Visualises the PnE optimisation problem
        :param img_rgb: The RGB image where the obs_gaussians were observed
        :param ax: The axis to draw the visualisation upon
        :param dual_quadratics: The dual quadratics that are projected onto the image
        :param cam_t_base: A 4x4 hom. matrix of the camera pose
        :param intrinsic_mtx: The 3x3 intrinsic camera matrix
        :param obs_gaussians_sigma_s: The cov. matrices of the observed gaussians
        :param obs_gaussians_mu_s: The means of the observed gaussians
        :param title: The title of the plot
        """
        # To projection
        proj_primal_conincals = project_dual_quadratics_to_primal_conicals_torch(
            dual_quadratics=dual_quadratics,
            cam_t_base=cam_t_base,
            intrinsic_mtx=intrinsic_mtx
        )
        proj_mu_s, proj_sigma_s = primal_conics_to_gaussian_ellipses_torch(proj_primal_conincals)
        proj_sigma_mu_s = torch.cat([proj_sigma_s, proj_mu_s.unsqueeze(-1)], dim = -1)
        proj_sigma_mu_s_np = proj_sigma_mu_s.detach().cpu().numpy()


        obs_sigma_mu_s = torch.cat([obs_gaussians_sigma_s, obs_gaussians_mu_s.unsqueeze(-1)], dim = -1)
        obs_sigma_mu_s_np = obs_sigma_mu_s.detach().cpu().numpy()

        ax.set_title(title)
        ax.imshow(img_rgb)

        n = obs_sigma_mu_s_np.shape[0]

        colors = plt.cm.jet(np.linspace(0,1, n))

        proj_ellipses = gaussian_ellipse_s_to_matplotlib_ellipse_s(
            gaussian_ellipse_s=proj_sigma_mu_s_np,
            colors=colors
        )

        obs_ellipses = gaussian_ellipse_s_to_matplotlib_ellipse_s(
            gaussian_ellipse_s=obs_sigma_mu_s_np,
            colors=colors,
            line_style="-"
        )

        for proj_e, obs_e in zip(proj_ellipses, obs_ellipses):
            ax.add_patch(proj_e)
            ax.add_patch(obs_e)


        ax.scatter(
            obs_sigma_mu_s_np[:,0,2],
            obs_sigma_mu_s_np[:,1,2],
            color=colors, s=5, alpha=0.8, marker = 'o')
        
        ax.scatter(
            proj_sigma_mu_s_np[:,0,2],
            proj_sigma_mu_s_np[:,1,2],
            color=colors, s=5, alpha=0.8, marker = 's')
        

        ax.quiver(
            proj_sigma_mu_s_np[:,0,2],
            proj_sigma_mu_s_np[:,1,2],
            obs_sigma_mu_s_np[:,0,2]-proj_sigma_mu_s_np[:,0,2], 
            obs_sigma_mu_s_np[:,1,2]-proj_sigma_mu_s_np[:,1,2],
            angles='xy', scale_units='xy', scale=1,
            color=colors,
            alpha=0.6,
            width=0.005
        )

        empty_lines = [
            mlines.Line2D([], [], color='black', linestyle='-',  linewidth=2, label='Observed'),
            mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Projected'),
        ]
        ax.legend(handles=empty_lines, loc='upper right')
        ax.axis('off')


def visualize_multiple_pne_optimizer_losses(ax: Axes, optimizers: list[PnEOptimizer]):
    rows = []

    for optimizer in optimizers:
        if optimizer.accumulated_losses is None:
            continue

        for run_idx, losses in enumerate(optimizer.accumulated_losses):
            for iteration, loss in enumerate(losses):
                rows.append({
                    "optimizer": str(optimizer),
                    "run": run_idx,
                    "iteration": iteration,
                    "loss": loss,
                })

    df = pd.DataFrame(rows)
    if not df.empty:
        sns.lineplot(data=df, x="iteration", y="loss", hue="optimizer", ax=ax)
        ax.set_yscale("log")
    