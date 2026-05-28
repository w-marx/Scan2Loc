from dataclasses import dataclass
import pypose as pp
from pypose.optim import LM
import torch, torch.nn as nn
from pypose.optim.solver import Cholesky, PINV
import numpy as np
from scipy.spatial.transform import Rotation
import sys, os
from ellipsoid_utilities_numpy import *


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utilities import *


class Reproj(nn.Module):

    def __init__(self, 
                cam_intrinsic:np.ndarray, 
                cam_t_base_quat_vec:np.ndarray,
                primal_conicals:np.ndarray,
                primal_quadratics:np.ndarray,
                ):
        super().__init__()
        """
        TODO
        """

        assert_intrinsic_mat(cam_intrinsic)
        assert cam_t_base_quat_vec.shape == (7,), f"wrong pose shape: {cam_t_base_quat_vec.shape} != (7,)"

        self.cam_t_base_se3 = pp.Parameter(pp.SE3(torch.tensor(cam_t_base_quat_vec, dtype = torch.float32)))

        self.register_buffer('intrinsic_mtx', torch.tensor(cam_intrinsic, dtype = torch.float32))

        self.register_buffer('primal_quadratics', torch.tensor(primal_quadratics, dtype = torch.float32))

        mu_s, sigma_s = Reproj.primal_conics_to_gaussian_ellipses(torch.tensor(primal_conicals, dtype = torch.float32))

        self.register_buffer('obs_gaussians_mu_s', mu_s)
        self.register_buffer('obs_gaussians_sigma_s', sigma_s)


    def wasserstein_distances_sq(
            mu1_s:torch.Tensor,
            sigma1_s:torch.Tensor,
            mu2_s:torch.Tensor,
            sigma2_s:torch.Tensor
        ):
        """
        Computes the squared wasserstein distances
        :param mu1_s: An Nx2 array of means
        :param sigma1_s: An Nx2x2 array of covariance matrices
        :param mu2_s: An Nx2 array of means
        :param sigma2_s: An Nx2x2 array of covariance matrices
        :returns an array of length N of the distances
        """
        mean_dist_sq = torch.sum((mu1_s-mu2_s)**2, axis = 1)
        sigma2_sqrt = Reproj.sqrtm_2x2(sigma2_s)
        cross = Reproj.sqrtm_2x2(sigma2_sqrt @ sigma1_s @ sigma2_sqrt)
        dist_sq = mean_dist_sq + torch.einsum('bii->b',sigma1_s+sigma2_s-2*cross)
        return dist_sq
    
    @staticmethod
    def sqrtm_2x2(matrices:torch.Tensor):
        """
        :param matrices: Nx2x2 matrix array
        :return Nx2x2 array with sqrt(m)
        """
        vals, vecs = torch.linalg.eigh(matrices)
        sqrt_vals = torch.sqrt(vals.clamp(min = 1e-9))
        return vecs @ torch.diag_embed(sqrt_vals) @ vecs.transpose(-2,1)


    def forward(self):
        #reproject points
        proj_primal_conincals = Reproj.project_primal_quadratics_to_primal_conicals(
            primal_quadratic=self.primal_quadratics,
            cam_t_base=self.cam_t_base_se3.matrix(),
            intrinsic_mtx=self.intrinsic_mtx
        )
        proj_mu_s, proj_sigma_s = self.primal_conics_to_gaussian_ellipses(proj_primal_conincals)

        distances = Reproj.wasserstein_distances_sq(
            mu1_s=proj_mu_s,
            sigma1_s=proj_sigma_s,
            mu2_s = self.obs_gaussians_mu_s,
            sigma2_s=self.obs_gaussians_sigma_s
        )
        return torch.sqrt(distances)
    

    @staticmethod
    def project_primal_quadratics_to_primal_conicals(
        primal_quadratic:torch.Tensor,
        cam_t_base:torch.Tensor,
        intrinsic_mtx:torch.Tensor
    ):
        """
        Takes Nx4x4 primal quadratics in the world frame
        and returns the Nx3x3 primal conics in the camera frame
        :param primal_quadratic: The primal quadratic as a Nx4x4 matrix
        :param cam_t_base: A 4x4 camera->base homogeneous transformation matrix
        :param intrinsic_mtx: The 3x3 intrinsic matrix of the camera

        :return: The 3x3 primal conic of the projected ellipse
        """

        dual_quadratic = torch.linalg.inv(primal_quadratic)
        P = cam_t_base[:3, :]
        KP = intrinsic_mtx @ P
        cam_dual_conic = torch.einsum('ik,nkl,jl->nij', KP, dual_quadratic, KP)
        #cam_dual_conic = (KP.unsqueeze(0) @ dual_quadratic) @ KP.T.unsqueeze(0)
        cam_primal_conic = torch.linalg.inv(cam_dual_conic)

        return cam_primal_conic

    @staticmethod
    def primal_conics_to_gaussian_ellipses(primal_conic:torch.Tensor)-> tuple[torch.Tensor, torch.Tensor]:
        """
        Takes Nx3x3 primal conics and returns normal distributions where the edge of the elipsoid is the Mahalanobis of one
        :param primal_conic: The Nx3x3 primal conics
        :return a tuple: Nx2 of the mu's and Nx2x2 of the sigma's
        """

        a = primal_conic[:, 0:2, 0:2]
        b = primal_conic[:, 0:2, 2]
        c = primal_conic[:, 2, 2]
        mu = -torch.linalg.solve(a, b)

        # normalization
        s = torch.einsum('ni,nij,nj->n', mu, a, mu) - c
        s = s.unsqueeze(-1).unsqueeze(-1)

        c_norm = primal_conic / s
        a_norm = c_norm[:, 0:2, 0:2]

        sigma = torch.linalg.inv(a_norm)

        return mu, sigma
    
    def visualize_2d(self, img_rgb:np.ndarray):
        import matplotlib.pyplot as plt


        # To projection
        proj_primal_conincals = Reproj.project_primal_quadratics_to_primal_conicals(
            primal_quadratic=self.primal_quadratics,
            cam_t_base=self.cam_t_base_se3.matrix(),
            intrinsic_mtx=self.intrinsic_mtx
        )
        proj_mu_s, proj_sigma_s = self.primal_conics_to_gaussian_ellipses(proj_primal_conincals)
        proj_sigma_mu_s = torch.cat([proj_sigma_s, proj_mu_s.unsqueeze(-1)], dim = -1)
        proj_sigma_mu_s_np = proj_sigma_mu_s.detach().cpu().numpy()


        obs_sigma_mu_s = torch.cat([self.obs_gaussians_sigma_s, self.obs_gaussians_mu_s.unsqueeze(-1)], dim = -1)
        obs_sigma_mu_s_np = obs_sigma_mu_s.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize = (12, 8))
        ax.set_title("PnE errors")
        ax.imshow(img_rgb)

        n = obs_sigma_mu_s_np.shape[0]

        colors = plt.cm.jet(np.linspace(0,1, n))

        proj_ellipses = gaussian_ellipse_s_to_matplotlib_ellipse_s(
            gaussian_ellipse_s=proj_sigma_mu_s_np,
            colors=colors
        )

        obs_ellipses = gaussian_ellipse_s_to_matplotlib_ellipse_s(
            gaussian_ellipse_s=obs_sigma_mu_s_np,
            colors=colors 
        )

        for proj_e, obs_e in zip(proj_ellipses, obs_ellipses):
            ax.add_patch(proj_e)
            ax.add_patch(obs_e)


        ax.scatter(
            obs_sigma_mu_s_np[:,0,2],
            obs_sigma_mu_s_np[:,1,2],
            color=colors, s=5, alpha=0.8, marker = 'o', label = 'Observed')
        
        ax.scatter(
            proj_sigma_mu_s_np[:,0,2],
            proj_sigma_mu_s_np[:,1,2],
            color=colors, s=5, alpha=0.8, marker = 's', label = 'Projected')
        

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

        ax.legend()

        plt.show()




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

@dataclass(frozen=True, kw_only=True)
class PnEOptimizerConfig:
    """
    Defines how the Perspective n Ellipsoids problem is optimized
    """
    lm_max_steps:int = 20
    lm_reject = 30
    lm_strat_up = 2.0
    lm_strat_down = 0.5
    solver = PINV
    convergence_threshold:float = 1e-6
    
    def __post_init__(self):
        assert self.lm_max_steps > 0
    

def optimize_pne(
        initial_cam_t_base:np.ndarray,
        primal_quadratics:np.ndarray,
        primal_conicals:np.ndarray,
        intrinsic_cam_mat:np.ndarray,
        config:PnEOptimizerConfig = PnEOptimizerConfig(),
        visualize_result:None | np.ndarray = False
)->np.ndarray:
    """
    :param initial_cam_t_base: 4x4 homogeneous matrix of the initial camera position
    :param points_3d: Nx3 array of points in the base_frame
    :param points_2d: Nx2 array of observed points in image coordinates (p3d_i corresponds to p2d_i) (with [wi, hi] indexing)
    :param lines_2d: Mx4 array of lines in the format [x1, y1, x2, y2] (l2d_i corresponds to l3d_i)
    :param lines_3d: Mx6 array of lines in the format [x1, y1, z1, x2, y2, z2]
    :param intrinsic_cam_mat: 3x3 Intrinsic camera matrix
    :param config: The config for the LM optimizer
    :param visualize_result: Wil visualize the optimization problem over the HxWx3-uint8 rgb image if wanted
    :param line_relevance: multiplier before the line error (point relevance = 1-line_relevance)
    """


    _ = assert_intrinsic_mat(intrinsic_cam_mat)
    _ = assert_homogeneous_mat(initial_cam_t_base, size=4)

    model = Reproj(
        cam_intrinsic=intrinsic_cam_mat,
        cam_t_base_quat_vec=hom_to_quat_vec(initial_cam_t_base),
        primal_conicals=primal_conicals,
        primal_quadratics=primal_quadratics
    )
    
    if visualize_result is not None:
        model.visualize_2d(visualize_result)

    inp = {}

    strategy = pp.optim.strategy.TrustRegion(up = config.lm_strat_up, down=config.lm_strat_down)
    opt = LM(model, solver=config.solver(), strategy=strategy, reject=config.lm_reject, sparse=False)
    losses = []
    for step in range(config.lm_max_steps):
        loss = opt.step(inp)
        print(f"step: {step}, loss: {loss}")
        losses.append(loss)
        if len(losses) > 2 and abs(losses[-1]-losses[-2]) < 1e-6:
            break

    final_cam_t_base = model.cam_t_base_se3.matrix().detach().cpu().numpy()

    if visualize_result is not None:
        model.visualize_2d(visualize_result)

    return final_cam_t_base