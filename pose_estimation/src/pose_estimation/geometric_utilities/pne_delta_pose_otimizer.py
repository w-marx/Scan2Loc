import pypose as pp
import torch, torch.nn as nn
from torch.optim import LBFGS
from torch.optim import Adam
from dataclasses import dataclass
from typing import Literal

from .pne_optimizer import *
from .pose_optimisation import *


@dataclass(frozen=True, kw_only=True)
class PnEDeltaPoseLBFGSOptimizerConfig:
    """
    Defines how the Perspective n Ellipsoids problem is optimized
    """
    learning_rate:float = 0.001
    max_itterations:int = 40
    stop_at_grad:float = 1e-9
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
        visualize_result:None | np.ndarray = None,
    )->np.ndarray:
        # Convert everything to torch & precompute
        torch_intrinsic = torch.tensor(intrinsic_cam_mat, dtype = torch.float32)
        dual_quadratics = torch.linalg.inv(torch.tensor(primal_quadratics, dtype = torch.float32))
        obs_mu_s, obs_sigma_s = primal_conics_to_gaussian_ellipses_torch(torch.tensor(primal_conicals, dtype = torch.float32))
        obs_sqrt_sigma_s = sqrtm_2x2_torch(obs_sigma_s)
        init_cam_t_base_torch = torch.tensor(initial_cam_t_base, dtype = torch.float32)


        x_i = torch.zeros(6, dtype = torch.float32, requires_grad = True)
        optimizer = LBFGS([x_i],
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
                cam_t_base=compute_pose_exp_se3(x_i, init_cam_t_base_torch),
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
            optimizer.zero_grad()
            loss = get_loss()
            loss.backward()
            losses.append(loss.item())
            return loss
        
        self.time_tracker.add_time_stamp("Initialisation")
        optimizer.step(closure)
        self.time_tracker.add_time_stamp("Optimisation")

        if visualize_result is not None:
            self.register_visualisation2(cam_t_base=compute_pose_exp_se3(x_i, init_cam_t_base_torch))
            self.register_visualisation_loss(losses)
            plt.show()
        
        return compute_pose_exp_se3(x_i, init_cam_t_base_torch).detach().cpu().numpy()
    



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
    delta_pose_mapping:Literal["euler", "se3_exp"] = "se3_exp"
    
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

        if config.delta_pose_mapping == "se3_exp":
            self.apply_delta_pose = compute_pose_exp_se3
        else:
            self.apply_delta_pose = compute_pose_euler


    def optimize_pne(
        self,
        initial_cam_t_base:np.ndarray,
        primal_quadratics:np.ndarray,
        primal_conicals:np.ndarray,
        intrinsic_cam_mat:np.ndarray,
        visualize_result:None | np.ndarray = None,
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
                cam_t_base=self.apply_delta_pose(x_i, init_cam_t_base_torch),
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
            self.register_visualisation2(cam_t_base=compute_pose_exp_se3(x_i, init_cam_t_base_torch))
            self.register_visualisation_loss(losses)
            plt.show()
        
        return compute_pose_exp_se3(x_i, init_cam_t_base_torch).detach().cpu().numpy()