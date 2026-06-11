import pypose as pp
from pypose.optim import LM
import torch, torch.nn as nn
from pypose.optim.solver import Cholesky, PINV
from dataclasses import dataclass
from typing import Literal

from shared.assertion_helpers import assert_intrinsic_mat, assert_homogeneous_mat

from .time_tracker import TimeTracker
from .pne_optimizer import *

class PyposePnEHelper(nn.Module):

    def __init__(self, 
                cam_intrinsic:np.ndarray, 
                cam_t_base_quat_vec:np.ndarray,
                primal_conicals:np.ndarray,
                primal_quadratics:np.ndarray,
                time_tracker:TimeTracker
                ):
        super().__init__()
        """
        TODO
        """
        self.time_tracker = time_tracker
        self.time_tracker.reset_elapsed_time()
        assert assert_intrinsic_mat(cam_intrinsic)
        assert cam_t_base_quat_vec.shape == (7,), f"wrong pose shape: {cam_t_base_quat_vec.shape} != (7,)"

        self.cam_t_base_se3 = pp.Parameter(pp.SE3(torch.tensor(cam_t_base_quat_vec, dtype = torch.float32)))

        self.register_buffer('intrinsic_mtx', torch.tensor(cam_intrinsic, dtype = torch.float32))

        self.register_buffer('dual_quadratics', torch.linalg.inv(torch.tensor(primal_quadratics, dtype = torch.float32)))

        mu_s, sigma_s = primal_conics_to_gaussian_ellipses_torch(torch.tensor(primal_conicals, dtype = torch.float32))
        self.register_buffer('obs_sigma_s_sqrt', sqrtm_2x2_torch(sigma_s))
        self.register_buffer('obs_gaussians_mu_s', mu_s)
        self.register_buffer('obs_gaussians_sigma_s', sigma_s)

        self.time_tracker.add_time_stamp("Initialisation")
        self.num_fw_calls = 0


    def forward(self):
        self.time_tracker.reset_elapsed_time()

        proj_primal_conincals = project_dual_quadratics_to_primal_conicals_torch(
            dual_quadratics=self.dual_quadratics,
            cam_t_base=self.cam_t_base_se3.matrix(),
            intrinsic_mtx=self.intrinsic_mtx
        )
        self.time_tracker.add_time_stamp("Project primal quadratics")

        proj_mu_s, proj_sigma_s = primal_conics_to_gaussian_ellipses_torch(proj_primal_conincals)

        self.time_tracker.add_time_stamp("Generate primal gaussians")

        distances = wasserstein_distances_sq_torch(
            mu1_s=proj_mu_s,
            sigma1_s=proj_sigma_s,
            mu2_s = self.obs_gaussians_mu_s,
            sigma2_s=self.obs_gaussians_sigma_s,
            sigma2_s_sqrt=self.obs_sigma_s_sqrt
        )
        self.time_tracker.add_time_stamp("Calculate squared wasserstein distances")
        self.time_tracker.add_time_stamp("everything")
        self.num_fw_calls += 1
        return torch.sqrt(distances)


@dataclass(frozen=True, kw_only=True)
class PyposePnEOptimizerConfig:
    """
    Defines how the Perspective n Ellipsoids problem is optimized
    """
    lm_max_steps:int = 40
    lm_reject:float = 30
    lm_strat_up:float = 3.0
    lm_strat_down:float = 0.5
    solver:Literal["PINV", "Cholesky"] = "PINV"
    convergence_threshold:float = 1e-6
    
    def __post_init__(self):
        assert self.lm_max_steps > 0

class PyposePNEOptimizer(PnEOptimizer):
    def __init__(
            self,
            config:PyposePnEOptimizerConfig = PyposePnEOptimizerConfig(),
            time_tracker:TimeTracker = TimeTracker(),
        )->None:
        self.config = config
        self.time_tracker = time_tracker
        self.optimize_pne_tt = TimeTracker()

        self.strategy = pp.optim.strategy.TrustRegion(up = self.config.lm_strat_up, down=self.config.lm_strat_down)

        solvers = {"PINV":PINV, "Cholesky":Cholesky}
        self.solver = solvers[config.solver]()
        self.number_fw_calls = []

    def optimize_pne(
        self,
        initial_cam_t_base:np.ndarray,
        primal_quadratics:np.ndarray,
        primal_conicals:np.ndarray,
        intrinsic_cam_mat:np.ndarray,
        visualize_result:None | np.ndarray = None,
    ):
        assert assert_homogeneous_mat(initial_cam_t_base, size=4)

        self.optimize_pne_tt.reset_elapsed_time()
        model = PyposePnEHelper(
            cam_intrinsic=intrinsic_cam_mat,
            cam_t_base_quat_vec=hom_to_quat_vec(initial_cam_t_base),
            primal_conicals=primal_conicals,
            primal_quadratics=primal_quadratics,
            time_tracker=self.time_tracker
        )
        self.optimize_pne_tt.add_time_stamp("Model creation")


        if visualize_result is not None:
            self.register_visualisation1(
                img_rgb=visualize_result, 
                dual_quadratics=model.dual_quadratics,
                cam_t_base=model.cam_t_base_se3.matrix(),
                obs_gaussians_mu_s=model.obs_gaussians_mu_s, 
                obs_gaussians_sigma_s=model.obs_gaussians_sigma_s,
                title="Before Optimisation", 
                intrinsic_mtx=model.intrinsic_mtx
            )

        inp = {}

        opt = LM(model, solver=self.solver, strategy=self.strategy, reject=self.config.lm_reject, sparse=False)

        self.optimize_pne_tt.add_time_stamp("Define opt")


        losses = []
        for step in range(self.config.lm_max_steps):
            loss = opt.step(inp)
            losses.append(loss)
            if len(losses) > 2 and abs(losses[-1]-losses[-2]) < 1e-6:
                break
        self.optimize_pne_tt.add_time_stamp("Optimisation loop")

        self.number_fw_calls.append(model.num_fw_calls)


        final_cam_t_base = model.cam_t_base_se3.matrix().detach().cpu().numpy()

        if visualize_result is not None:
            self.register_visualisation2(
                cam_t_base=model.cam_t_base_se3.matrix()
            )
            self.register_visualisation_loss(losses=losses)
            plt.show()
        
        
        self.optimize_pne_tt.add_time_stamp("Cleanup")
        return final_cam_t_base