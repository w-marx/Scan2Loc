from dataclasses import dataclass
import pypose as pp
from pypose.optim import LM
import torch, torch.nn as nn
from pypose.optim.solver import Cholesky, PINV
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

from shared.assertion_helpers import assert_intrinsic_mat, assert_homogeneous_mat

from .line_utilities import project_point_onto_line_slow


class Reproj(nn.Module):

    def __init__(self, 
                cam_intrinsic:np.ndarray, 
                cam_t_base_quat_vec:np.ndarray,
                points_2d:np.ndarray, 
                points_3d:np.ndarray,
                lines_2d:np.ndarray,
                lines_3d:np.ndarray,
                line_relevance:float
                ):
        super().__init__()
        """
        Creates an optimization problem, that minimizes the euclidian distance between the 
        projected 3d points and 2d points and the distance between the endpoints of the projected
        3d line endpoints and infinite 2d lines.
        (matches lines & points by indices)

        :param cam_intrinsic: 3x3 intrinsic camera matrix (numpy array)
        :param cam_t_base_quat_vec: length 7 array of the camera pose
        :param points_3d: Nx3 point_cloud of xyz-points in the base-frame
        :param lines_3d: Nx6 array of line segments in the base_frame of the structure: [[x1, y1, z1, x2, y2, z2], ...]
        :param lines_2d: Nx4 array of line segments in the base_frame of the structure: [[x1, y1, x2, y2], ...]
        :param line_relevance: Value in [0,1], 0-> only point error is minimized, 1 -> only line error is minimized
        """

        n_points = points_2d.shape[0]
        n_lines = lines_2d.shape[0]
        n_points_combined = n_points + 2 * n_lines

        assert cam_intrinsic.shape == (3,3), f"wrong intrinsics shape: {cam_intrinsic.shape} != (3,3)"
        assert cam_t_base_quat_vec.shape == (7,), f"wrong pose shape: {cam_t_base_quat_vec.shape} != (7,)"

        assert points_3d.ndim == 2 and points_3d.shape[-1] == 3, f"3d point shape: {points_3d.shape} != (N,3)"
        assert points_2d.ndim == 2 and points_2d.shape[-1] == 2, f"3d point shape: {points_2d.shape} != (N,2)"
        assert points_3d.shape[0] == points_2d.shape[0], f"lengths of 2d and 3d points dont match: {points_3d.shape}, {points_2d.shape}"

        assert lines_3d.ndim == 2 and lines_3d.shape[-1] == 6, f"3d line shape: {lines_3d.shape} != (N,6)"
        assert lines_2d.ndim == 2 and lines_2d.shape[-1] == 4, f"2d line shape: {lines_2d.shape} != (N,4)"
        assert lines_3d.shape[0] == lines_2d.shape[0], f"lengths of 2d and 3d lines dont match: {lines_3d.shape}, {lines_2d.shape}"

        assert 0 <= line_relevance <= 1, f"line relevance {line_relevance} not in [0,1]"

        self.cam_t_base_se3 = pp.Parameter(pp.SE3(torch.tensor(cam_t_base_quat_vec, dtype = torch.float32)))

        self.register_buffer('intrinsics', torch.tensor([cam_intrinsic[0,0], cam_intrinsic[1,1], cam_intrinsic[0,2], cam_intrinsic[1,2]], dtype = torch.float32))


        self.n_points = n_points
        self.n_lines = n_lines
        self.n_points_combined = n_points_combined

        self.register_buffer('lines_3d', torch.tensor(lines_3d, dtype = torch.float32))

        points_3d_t = torch.tensor(points_3d, dtype = torch.float32)
        lines_3d_t = torch.tensor(lines_3d, dtype = torch.float32)

        self.register_buffer('points_3d_combined', torch.concat([points_3d_t, lines_3d_t.reshape(-1,3)]))

        self.register_buffer('observed_points_2d', torch.tensor(points_2d, dtype = torch.float32))
        self.register_buffer('_point_error_multiplier', torch.tensor((1-line_relevance)/n_points, dtype = torch.float32))

        self.register_buffer('_error_buffer', torch.empty(n_points +n_lines,2, dtype = torch.float32))

        lines = torch.tensor(lines_2d, dtype = torch.float32)
        x1, y1, x2, y2 = lines[:,0], lines[:,1], lines[:,2], lines[:,3]

        dx = x2 - x1
        dy = y2 - y1

        self.register_buffer('obs_lines_dx', dx)
        self.register_buffer('obs_lines_dy', dy)
        self.register_buffer('lines_dist_norm', line_relevance/(torch.sqrt(dx**2 + dy**2+ 1e-8).clamp(min=1e-6)*n_lines))
        self.register_buffer('obs_lines_c', x2*y1 - y2*x1)



    def forward(self):
        #reproject points
        proj_points = self.project_points()

        # Point errors
        self._error_buffer[:self.n_points] = (proj_points[:self.n_points]-self.observed_points_2d) * self._point_error_multiplier
    
        # Line errors
        proj_lines_endpoints = proj_points[self.n_points:].reshape(self.n_lines,2, 2)
        px = proj_lines_endpoints[..., 0]
        py = proj_lines_endpoints[..., 1]
        line_distances = (px * self.obs_lines_dy[:,None] - py * self.obs_lines_dx[:,None] + self.obs_lines_c[:,None]) * self.lines_dist_norm[:, None]
        self._error_buffer[self.n_points:] = line_distances

        return self._error_buffer
    
    def project_points(self):
        fx, fy, cx, cy = self.intrinsics

        cp = self.cam_t_base_se3.Act(self.points_3d_combined)

        n = cp[..., :2]/cp[..., [2]]

        u = n[..., 0:1]*fx+cx
        v = n[..., 1:2]*fy+cy
        return torch.cat([u, v], dim=-1)

    
    def visualize_2d(self, img_rgb:np.ndarray, observed_lines_2d:np.ndarray):

        proj_points = self.project_points().detach().cpu().numpy()
        obs_points = self.observed_points_2d.detach().cpu().numpy()

        fig, ax = plt.subplots(figsize = (12, 8))
        ax.set_title("PnPL errors")
        ax.imshow(img_rgb)


        colors = plt.cm.jet(np.linspace(0,1, self.n_points))
        ax.scatter(proj_points[:self.n_points, 0], proj_points[:self.n_points, 1], color=colors, s=5, alpha=0.8, marker = 'o', label = 'Projected')
        ax.scatter(obs_points[:, 0], obs_points[:, 1], color=colors, s=5, alpha=0.8, marker = 's', label = 'Observed')

        ax.quiver(
            proj_points[:self.n_points, 0], proj_points[:self.n_points, 1],
            obs_points[:, 0]-proj_points[:self.n_points, 0], obs_points[:, 1]-proj_points[:self.n_points, 1],
            angles='xy', scale_units='xy', scale=1,
            color=colors,
            alpha=0.6,
            width=0.005
        )

        colors_lines = plt.cm.jet(np.linspace(0,1, self.n_lines))
        for i,(x1, y1, x2, y2) in enumerate(observed_lines_2d):
            ax.axline((x1, y1), (x2, y2), color=colors_lines[i], linestyle=':', linewidth=1)
            ax.plot([x1, x2], [y1, y2], color=colors_lines[i], linewidth=1)


        line_seg_points = proj_points[self.n_points:, :].reshape(self.n_lines, 4)
        for i,((ox1, oy1, ox2, oy2), (x1, y1, x2, y2)) in enumerate(zip(observed_lines_2d, line_seg_points)):
            ax.scatter([x1, x2], [y1, y2], color=colors_lines[i], s=10, alpha=0.8, marker = 'x')

            px1, py1 = project_point_onto_line_slow(x1, y1, ox1, oy1, ox2, oy2)
            px2, py2 = project_point_onto_line_slow(x2, y2, ox1, oy1, ox2, oy2)

            ax.quiver([x1, x2], [y1, y2],
                        [px1 - x1, px2 - x2], [py1 - y1, py2 - y2], 
                        angles='xy', scale_units='xy', scale=1,
                        color=colors_lines[i], alpha=0.8, width=0.005)
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
class PnPLOptimizerConfig:
    """
    Discribes a line merging pass, that merges the lines and then removes
    lines with line_length < min_line_length
    """
    lm_max_steps:int = 20
    lm_reject = 30
    lm_strat_up = 2.0
    lm_strat_down = 0.5
    solver = PINV
    convergence_threshold:float = 1e-6
    line_relevance:float = 0.2
    

    def __post_init__(self):
        assert self.lm_max_steps > 0
        assert 0 <= self.line_relevance <= 1
    

def optimize_pnpl(
        initial_cam_t_base:np.ndarray,
        points_3d:np.ndarray,
        points_2d:np.ndarray,
        lines_2d:np.ndarray,
        lines_3d:np.ndarray,
        intrinsic_cam_mat:np.ndarray,
        config:PnPLOptimizerConfig = PnPLOptimizerConfig(),
        visualize_result:None | np.ndarray = None
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

    assert points_3d.shape[0] == points_2d.shape[0]
    assert points_2d.ndim == 2 and points_2d.shape[-1] == 2
    assert points_3d.ndim == 2 and points_3d.shape[-1] == 3


    assert lines_2d.shape[0] == lines_3d.shape[0]
    assert lines_2d.ndim == 2 and lines_2d.shape[-1] == 4
    assert lines_3d.ndim == 2 and lines_3d.shape[-1] == 6

    assert assert_intrinsic_mat(intrinsic_cam_mat)
    assert assert_homogeneous_mat(initial_cam_t_base)

    model = Reproj(
        cam_intrinsic= intrinsic_cam_mat,
        cam_t_base_quat_vec=hom_to_quat_vec(initial_cam_t_base),
        points_3d=points_3d,
        lines_3d=lines_3d,
        points_2d=points_2d,
        lines_2d=lines_2d,
        line_relevance=config.line_relevance
    )

    inp = {}

    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5)
    opt = LM(model, solver=config.solver(), strategy=strategy, reject=config.lm_reject, sparse=False)
    losses = []
    for step in range(config.lm_max_steps):
        loss = opt.step(inp)
        losses.append(loss)
        if len(losses) > 2 and abs(losses[-1]-losses[-2]) < 1e-6:
            break

    final_cam_t_base = model.cam_t_base_se3.matrix().detach().cpu().numpy()

    if visualize_result is not None:
        model.visualize_2d(visualize_result, lines_2d)

    return final_cam_t_base