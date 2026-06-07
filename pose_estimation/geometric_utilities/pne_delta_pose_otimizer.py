import pypose as pp
import torch, torch.nn as nn
from torch.optim import LBFGS
from torch.optim import Adam
from dataclasses import dataclass
from typing import Literal

from .pne_optimizer import *

def d_rot(w):
    wx, wy, wz = w

    z = torch.zeros((), dtype=w.dtype, device=w.device)

    return torch.stack([
        torch.stack([z,  -wz,  wy]),
        torch.stack([wz,  z,  -wx]),
        torch.stack([-wy, wx,  z]),
    ])

def exp_se3(S:torch.Tensor):
    """
    Takes an array: S = (Sw, Sv)
    And returns exp([S]*theta)
    :param s: An array: [w1, w2, w3, s1, s2, s3]
    :return: 4x4 hom. matrix: exp([S]*theta)
    """
    Sw, Sv = S[:3], S[3:]

    # theta = |Sw|
    theta = torch.norm(Sw)
    r = torch.eye(3, device=S.device, dtype=S.dtype)


    # |Sw| = 0
    # exp([S]theta) = | I   Sv*theta    |
    #                 | 0       1       |
    if theta < 1e-8:
        r += d_rot(Sw)
        V = r
    # |Sw| = 1:
    # exp([S]theta) = | exp[Sw*theta]   (I*theta + (1-c(theta))*Sw + (theta-s(theta)*Sw^2))Sv  |
    #                 | 0                                                               1      |

    else:
        K = d_rot(Sw / theta)
        r += torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)

        V = (
            r
            + (1 - torch.cos(theta)) / theta * K
            + (theta - torch.sin(theta)) / (theta**2) * (K @ K)
        )

    T = torch.eye(4, device=S.device, dtype=S.dtype)
    T[:3, :3] = r
    T[:3, 3] = V @ Sv
    return T

def compute_pose(x_i, cam_t_base):
    return exp_se3(x_i) @ cam_t_base


@dataclass(frozen=True, kw_only=True)
class PnEDeltaPoseLBFGSOptimizerConfig:
    """
    Defines how the Perspective n Ellipsoids problem is optimized
    """
    learning_rate:float = 0.001
    max_itterations:int = 40
    stop_at_grad:float = 1e-6
    convergence_threshold:float = 1e-6
    history_size:int = 10
    line_search_function:Literal["strong_wolfe"] = 'strong_wolfe'
    
    def __post_init__(self):
        pass

class PnEDeltaPoseLBFGSOptimizer(PnEOptimizer):
    def __init__(
            self, 
            config:PnEDeltaPoseLBFGSOptimizerConfig = PnEDeltaPoseLBFGSOptimizerConfig(),
            time_tracker:TimeTracker = TimeTracker()
        ) -> None:
        super().__init__()
        self.time_tracker = time_tracker
        self.config = config

    def optimize_pne(
        self,
        initial_cam_t_base:np.ndarray,
        primal_quadratics:np.ndarray,
        primal_conicals:np.ndarray,
        intrinsic_cam_mat:np.ndarray,
        visualize_result:None | np.ndarray = None
    ):
        # Convert everything to torch & precompute
        torch_intrinsic = torch.tensor(intrinsic_cam_mat, dtype = torch.float32)
        dual_quadratics = torch.linalg.inv(torch.tensor(primal_quadratics, dtype = torch.float32))
        obs_mu_s, obs_sigma_s = primal_conics_to_gaussian_ellipses_torch(torch.tensor(primal_conicals, dtype = torch.float32))
        obs_sqrt_sigma_s = sqrtm_2x2_torch(obs_sigma_s)
        init_cam_t_base_torch = torch.tensor(initial_cam_t_base, dtype = torch.float32)


        x_i = torch.zeros(6, dtype = torch.float32, requires_grad = True)
        self.optimizer = LBFGS([x_i], 
                               lr=self.config.learning_rate, 
                               max_iter=self.config.max_itterations, 
                               tolerance_grad=self.config.stop_at_grad, 
                               tolerance_change=self.config.convergence_threshold,
                               history_size=self.config.history_size, 
                               line_search_fn=self.config.line_search_function)

        if visualize_result is not None:
            self.register_visualisation1(img_rgb=visualize_result, dual_quadratics=dual_quadratics,
                cam_t_base=init_cam_t_base_torch,obs_gaussians_mu_s=obs_mu_s, obs_gaussians_sigma_s=obs_sigma_s,
                title="Before Optimisation", intrinsic_mtx=torch_intrinsic
            )

        def get_loss():
            proj_primal_conicals = project_dual_quadratics_to_primal_conicals_torch(
                dual_quadratics=dual_quadratics,
                cam_t_base=compute_pose(x_i, init_cam_t_base_torch),
                intrinsic_mtx=torch_intrinsic
            )
            proj_mu_s, proj_sigma_s = primal_conics_to_gaussian_ellipses_torch(primal_conic_s=proj_primal_conicals)
            dist = wasserstein_distances_sq_torch(
                mu1_s=proj_mu_s, sigma1_s=proj_sigma_s,
                mu2_s=obs_mu_s, sigma2_s=obs_sigma_s, sigma2_s_sqrt=obs_sqrt_sigma_s
            )
            return dist.sum()
        
        losses = []
        def closure():
            self.optimizer.zero_grad()
            loss = get_loss()
            loss.backward()
            losses.append(loss.item())
            return loss
        
        self.time_tracker.add_time_stamp("Initialisation")
        self.optimizer.step(closure)
        self.time_tracker.add_time_stamp("Optimisation")

        if visualize_result is not None:
            self.register_visualisation2(cam_t_base=compute_pose(x_i, init_cam_t_base_torch))
            self.register_visualisation_loss(losses)
            plt.show()
        
        return compute_pose(x_i, init_cam_t_base_torch).detach().cpu().numpy()
    



@dataclass(frozen=True, kw_only=True)
class PnEDeltaPoseAdamOptimizerConfig:
    """
    Defines how the Perspective n Ellipsoids problem is optimized
    """
    learning_rate:float = 0.001
    betas:tuple[float, float] = (0.9, 0.999)
    max_itterations:int = 100
    convergence_threshold:float = 1e-6
    eps:float = 1e-8
    
    def __post_init__(self):
        pass

class PnEDeltaPoseAdamOptimizer(PnEOptimizer):
    def __init__(
            self, 
            config:PnEDeltaPoseAdamOptimizerConfig = PnEDeltaPoseAdamOptimizerConfig(),
            time_tracker:TimeTracker = TimeTracker()
        ) -> None:
        super().__init__()
        self.time_tracker = time_tracker
        self.config = config

    def optimize_pne(
        self,
        initial_cam_t_base:np.ndarray,
        primal_quadratics:np.ndarray,
        primal_conicals:np.ndarray,
        intrinsic_cam_mat:np.ndarray,
        visualize_result:None | np.ndarray = None
    ):
        # Convert everything to torch & precompute
        torch_intrinsic = torch.tensor(intrinsic_cam_mat, dtype = torch.float32)
        dual_quadratics = torch.linalg.inv(torch.tensor(primal_quadratics, dtype = torch.float32))
        obs_mu_s, obs_sigma_s = primal_conics_to_gaussian_ellipses_torch(torch.tensor(primal_conicals, dtype = torch.float32))
        obs_sqrt_sigma_s = sqrtm_2x2_torch(obs_sigma_s)
        init_cam_t_base_torch = torch.tensor(initial_cam_t_base, dtype = torch.float32)
        x_i = torch.zeros(6, dtype = torch.float32, requires_grad = True)

        if visualize_result is not None:
            self.register_visualisation1(img_rgb=visualize_result, dual_quadratics=dual_quadratics,
                cam_t_base=init_cam_t_base_torch,obs_gaussians_mu_s=obs_mu_s, obs_gaussians_sigma_s=obs_sigma_s,
                title="Before Optimisation", intrinsic_mtx=torch_intrinsic
            )

        def get_loss():
            proj_primal_conicals = project_dual_quadratics_to_primal_conicals_torch(
                dual_quadratics=dual_quadratics,
                cam_t_base=compute_pose(x_i, init_cam_t_base_torch),
                intrinsic_mtx=torch_intrinsic
            )
            proj_mu_s, proj_sigma_s = primal_conics_to_gaussian_ellipses_torch(primal_conic_s=proj_primal_conicals)
            dist = wasserstein_distances_sq_torch(
                mu1_s=proj_mu_s, sigma1_s=proj_sigma_s,
                mu2_s=obs_mu_s, sigma2_s=obs_sigma_s, sigma2_s_sqrt=obs_sqrt_sigma_s
            )
            return dist.sum()
        
        optimizer = torch.optim.Adam(
            [x_i],
            lr = self.config.learning_rate,
            betas = self.config.betas,
            eps = self.config.eps
        )
        

        self.time_tracker.add_time_stamp("Initialisation")
        losses = []
        for _ in range(self.config.max_itterations):
            optimizer.zero_grad()
            loss = get_loss()
            loss.backward()

            optimizer.step()
            losses.append(loss.item())

            if len(losses) > 2 and abs(losses[-1]-losses[-2]) <= self.config.convergence_threshold:
                break
        self.time_tracker.add_time_stamp("Optimisation")

        if visualize_result is not None:
            self.register_visualisation2(cam_t_base=compute_pose(x_i, init_cam_t_base_torch))
            self.register_visualisation_loss(losses)
            plt.show()
        
        return compute_pose(x_i, init_cam_t_base_torch).detach().cpu().numpy()