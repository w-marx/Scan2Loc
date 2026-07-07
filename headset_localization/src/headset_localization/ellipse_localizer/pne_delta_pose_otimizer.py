import torch
from torch.optim import LBFGS, Adam
from dataclasses import dataclass
from typing import Literal, Callable
import numpy as np
import matplotlib.pyplot as plt

from .pne_optimizer import PnEOptimizer, primal_conics_to_gaussian_ellipses_torch, sqrtm_2x2_torch, full_error_calculation, full_error_calculation_compiled
from ..utilities.time_tracker import TimeTracker, TimeLabels
from ..utilities.pose_optimisation import compute_pose_exp_se3, compute_pose_euler, AdamConfig


def get_compiled_ellipsoid_loss_fn(
        pose_calculation:Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    )->Callable[[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:

    def loss_fn(
        dual_quadratics:torch.Tensor, 
        x_i:torch.Tensor, 
        init_cam_t_base:torch.Tensor,
        intrinsic_mtx:torch.Tensor, 
        obs_mu_s:torch.Tensor, 
        obs_sigma_s:torch.Tensor, 
        obs_sqrt_sigma_s:torch.Tensor
    ):
        return full_error_calculation(
                dual_quadratics=dual_quadratics,
                cam_t_base=pose_calculation(x_i, init_cam_t_base),
                intrinsic_mtx=intrinsic_mtx,
                obs_gaussians_mu_s = obs_mu_s,
                obs_gaussians_sigma_s = obs_sigma_s,
                obs_sigma_s_sqrt = obs_sqrt_sigma_s
            ).sum()

    return torch.compile(loss_fn)

@dataclass(frozen=True, kw_only=True)
class PnEDeltaPoseLBFGSOptimizerConfig:
    """
    Defines how the Perspective n Ellipsoids problem is optimized
    """
    learning_rate:float = 0.0001
    max_itterations:int = 40
    stop_at_grad:float = 1e-9
    convergence_threshold:float = 1e-6
    history_size:int = 10
    line_search_function:Literal["strong_wolfe"] = 'strong_wolfe'
    device:Literal['cpu', 'cuda'] = 'cpu'
    
    def __post_init__(self):
        pass


class PnEDeltaPoseLBFGSOptimizer(PnEOptimizer):
    def __init__(
            self, 
            config:PnEDeltaPoseLBFGSOptimizerConfig = PnEDeltaPoseLBFGSOptimizerConfig(),
            delta_pose_mapping:Literal["euler", "se3_exp"] = "se3_exp",
            time_tracker:TimeTracker = TimeTracker(),
            accumulate_losses:bool = False
        ) -> None:
        super().__init__(accumulate_losses = accumulate_losses)
        self.time_tracker = time_tracker
        self.config = config
        self.device = torch.device(config.device)

        self.delta_pose_mapping_str = delta_pose_mapping

        if delta_pose_mapping == "se3_exp":
            self.apply_delta_pose = compute_pose_exp_se3
        else:
            self.apply_delta_pose = compute_pose_euler

        self.loss_fn = get_compiled_ellipsoid_loss_fn(self.apply_delta_pose)



    def __str__(self):
        return f"PnE LBFGS on {self.device} w lr: {self.config.learning_rate:.5f} mx steps: {self.config.max_itterations}"


    def optimize_pne(
        self,
        initial_cam_t_base:np.ndarray,
        primal_quadratics:np.ndarray,
        primal_conicals:np.ndarray,
        intrinsic_cam_mat:np.ndarray,
        visualize_result:None | np.ndarray = None,
    )->np.ndarray:
        # Convert everything to torch & precompute
        torch_intrinsic = torch.tensor(intrinsic_cam_mat, dtype = torch.float32, device=self.device)
        dual_quadratics = torch.linalg.inv(torch.tensor(primal_quadratics, dtype = torch.float32, device=self.device))
        obs_mu_s, obs_sigma_s = primal_conics_to_gaussian_ellipses_torch(torch.tensor(primal_conicals, dtype = torch.float32, device=self.device))
        obs_sqrt_sigma_s = sqrtm_2x2_torch(obs_sigma_s)
        init_cam_t_base_torch = torch.tensor(initial_cam_t_base, dtype = torch.float32, device=self.device)


        x_i = torch.zeros(6, dtype = torch.float32, requires_grad = True, device=self.device)
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

        def get_loss()->torch.Tensor:            
            return self.loss_fn(
                dual_quadratics=dual_quadratics,
                x_i=x_i,
                init_cam_t_base=init_cam_t_base_torch,
                intrinsic_mtx=torch_intrinsic,
                obs_mu_s=obs_mu_s,
                obs_sigma_s=obs_sigma_s,
                obs_sqrt_sigma_s=obs_sqrt_sigma_s
            )
        
        losses = []
        def closure():
            optimizer.zero_grad()
            loss = get_loss()
            loss.backward()
            losses.append(loss.item())
            return loss
        
        if self.accumulated_losses is not None:
            self.accumulated_losses.append(losses)
        
        self.time_tracker.add_time_stamp("Initialisation")
        optimizer.step(closure)
        self.time_tracker.add_time_stamp("Optimisation")

        if visualize_result is not None:
            self.register_visualisation2(cam_t_base=self.apply_delta_pose(x_i, init_cam_t_base_torch))
            self.register_visualisation_loss(losses)
            plt.show()
        
        return self.apply_delta_pose(x_i, init_cam_t_base_torch).detach().cpu().numpy()
    


class PnEDeltaPoseAdamOptimizer(PnEOptimizer):
    def __init__(
            self, 
            adam_cfg:AdamConfig = AdamConfig(learning_rate=0.001, max_itterations=100),
            delta_pose_mapping:Literal["euler", "se3_exp"] = "se3_exp",
            device:Literal['cuda', 'cpu'] = 'cpu',
            time_tracker:TimeTracker = TimeTracker(),
            accumulate_losses:bool = False
        ) -> None:
        super().__init__(accumulate_losses=accumulate_losses)
        self.time_tracker = time_tracker
        self.adam_cfg = adam_cfg
        self.device = torch.device(device)
        self.delta_pose_mapping_str = delta_pose_mapping

        if delta_pose_mapping == "se3_exp":
            self.apply_delta_pose = compute_pose_exp_se3
        else:
            self.apply_delta_pose = compute_pose_euler

        self.loss_fn = get_compiled_ellipsoid_loss_fn(self.apply_delta_pose)


    def __str__(self):
        return f"PnE Adam on {self.device}, mappint: {self.delta_pose_mapping_str} w lr: {self.adam_cfg.learning_rate:.5f} mx steps: {self.adam_cfg.max_itterations}"


    def optimize_pne(
        self,
        initial_cam_t_base:np.ndarray,
        primal_quadratics:np.ndarray,
        primal_conicals:np.ndarray,
        intrinsic_cam_mat:np.ndarray,
        visualize_result:None | np.ndarray = None,
    ):
        # Convert everything to torch & precompute
        torch_intrinsic = torch.tensor(intrinsic_cam_mat, dtype = torch.float32, device=self.device)
        dual_quadratics = torch.linalg.inv(torch.tensor(primal_quadratics, dtype = torch.float32, device=self.device))
        obs_mu_s, obs_sigma_s = primal_conics_to_gaussian_ellipses_torch(torch.tensor(primal_conicals, dtype = torch.float32, device=self.device))
        obs_sqrt_sigma_s = sqrtm_2x2_torch(obs_sigma_s)
        init_cam_t_base_torch = torch.tensor(initial_cam_t_base, dtype = torch.float32, device=self.device)
        x_i = torch.zeros(6, dtype = torch.float32, requires_grad = True, device=self.device)

        if visualize_result is not None:
            self.register_visualisation1(img_rgb=visualize_result, dual_quadratics=dual_quadratics,
                cam_t_base=init_cam_t_base_torch,obs_gaussians_mu_s=obs_mu_s, obs_gaussians_sigma_s=obs_sigma_s,
                title="Before Optimisation", intrinsic_mtx=torch_intrinsic
            )

        def get_loss()->torch.Tensor:
            return self.loss_fn(
                dual_quadratics=dual_quadratics,
                x_i=x_i,
                init_cam_t_base=init_cam_t_base_torch,
                intrinsic_mtx=torch_intrinsic,
                obs_mu_s=obs_mu_s,
                obs_sigma_s=obs_sigma_s,
                obs_sqrt_sigma_s=obs_sqrt_sigma_s
            )
        
        optimizer = Adam(
            [x_i],
            lr = self.adam_cfg.learning_rate,
            betas = self.adam_cfg.betas,
            eps = self.adam_cfg.eps
        )
        

        self.time_tracker.add_time_stamp("Initialisation")
        losses = []
        for _ in range(self.adam_cfg.max_itterations):
            optimizer.zero_grad()
            loss = get_loss()
            loss.backward()

            optimizer.step()
            losses.append(loss.item())

            if len(losses) > 2 and abs(losses[-1]-losses[-2]) <= self.adam_cfg.convergence_threshold:
                break
        self.time_tracker.add_time_stamp("Optimisation")

        if self.accumulated_losses is not None:
            self.accumulated_losses.append(losses)

        if visualize_result is not None:
            self.register_visualisation2(cam_t_base=self.apply_delta_pose(x_i, init_cam_t_base_torch))
            self.register_visualisation_loss(losses)
            plt.show()
        
        return self.apply_delta_pose(x_i, init_cam_t_base_torch).detach().cpu().numpy()