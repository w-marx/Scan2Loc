import torch
from dataclasses import dataclass
from typing import Literal
import matplotlib as plt

from .pne_optimizer import *


def euler_to_matrix(angles):
    """
    angles: (3,) tensor with requires_grad=True
    return: (3,3) rotation matrix, fully differentiable w.r.t. angles
    """
    rx, ry, rz = angles
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)

    one = torch.ones((), device=angles.device)
    zero = torch.zeros((), device=angles.device)

    Rx = torch.stack([
        torch.stack([one,  zero,  zero]),
        torch.stack([zero,  cx,  -sx]),
        torch.stack([zero,  sx,   cx])
    ])
    Ry = torch.stack([
        torch.stack([ cy,  zero,  sy]),
        torch.stack([zero,  one,  zero]),
        torch.stack([-sy,  zero,  cy])
    ])
    Rz = torch.stack([
        torch.stack([ cz, -sz, zero]),
        torch.stack([ sz,  cz, zero]),
        torch.stack([zero, zero, one])
    ])

    return Rz @ Ry @ Rx

def build_grad_hom_matrix(x_i, hom_init) -> torch.Tensor:
    angles = x_i[:3]
    t = x_i[3:]

    r_delta = euler_to_matrix(angles)

    delta_pose = torch.eye(4, dtype=x_i.dtype, device=x_i.device)
    delta_pose[:3, :3] = r_delta
    delta_pose[:3, 3] = t

    return delta_pose @ hom_init

@dataclass(frozen=True, kw_only=True)
class PnEEulerAnglesOptimizerConfig:
    """
    Defines how the Perspective n Ellipsoids problem is optimized
    """
    learning_rate: float = 0.001
    betas: tuple[float, float] = (0.9, 0.999)
    max_itterations: int = 100
    convergence_threshold: float = 1e-6
    eps: float = 1e-8

    def __post_init__(self):
        pass


class PnEEulerAnglesOptimizer(PnEOptimizer):
    def __init__(
            self,
            config: PnEEulerAnglesOptimizerConfig = PnEEulerAnglesOptimizerConfig(),
            time_tracker: TimeTracker = TimeTracker()
    ) -> None:
        super().__init__()
        self.time_tracker = time_tracker
        self.config = config

    def optimize_pne(
            self,
            initial_cam_t_base: np.ndarray,
            primal_quadratics: np.ndarray,
            primal_conicals: np.ndarray,
            intrinsic_cam_mat: np.ndarray,
            visualize_result: None | np.ndarray = None,
    ):
        # Convert everything to torch & precompute
        torch_intrinsic = torch.tensor(intrinsic_cam_mat, dtype=torch.float32)
        dual_quadratics = torch.linalg.inv(torch.tensor(primal_quadratics, dtype=torch.float32))
        obs_mu_s, obs_sigma_s = primal_conics_to_gaussian_ellipses_torch(
            torch.tensor(primal_conicals, dtype=torch.float32))
        obs_sqrt_sigma_s = sqrtm_2x2_torch(obs_sigma_s)
        init_cam_t_base_torch = torch.tensor(initial_cam_t_base, dtype=torch.float32)
        x_i = torch.zeros(6, dtype=torch.float32, requires_grad=True)

        if visualize_result is not None:
            self.register_visualisation1(img_rgb=visualize_result, dual_quadratics=dual_quadratics,
                                         cam_t_base=init_cam_t_base_torch, obs_gaussians_mu_s=obs_mu_s,
                                         obs_gaussians_sigma_s=obs_sigma_s,
                                         title="Before Optimisation", intrinsic_mtx=torch_intrinsic
                                         )

        def get_loss():
            proj_primal_conicals = project_dual_quadratics_to_primal_conicals_torch(
                dual_quadratics=dual_quadratics,
                cam_t_base=build_grad_hom_matrix(x_i, init_cam_t_base_torch),
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
            lr=self.config.learning_rate,
            betas=self.config.betas,
            eps=self.config.eps
        )

        self.time_tracker.add_time_stamp("Initialisation")
        losses = []
        for _ in range(self.config.max_itterations):
            optimizer.zero_grad()
            loss = get_loss()
            loss.backward()

            optimizer.step()
            losses.append(loss.item())

            if len(losses) > 2 and abs(losses[-1] - losses[-2]) <= self.config.convergence_threshold:
                break
        self.time_tracker.add_time_stamp("Optimisation")

        if visualize_result is not None:
            self.register_visualisation2(cam_t_base=build_grad_hom_matrix(x_i, init_cam_t_base_torch))
            self.register_visualisation_loss(losses)
            plt.show()

        return build_grad_hom_matrix(x_i, init_cam_t_base_torch).detach().cpu().numpy()