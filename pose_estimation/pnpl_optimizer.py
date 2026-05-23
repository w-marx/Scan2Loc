import pypose as pp
from pypose.optim import LM
import torch, torch.nn as nn
from pypose.optim.solver import Cholesky
import numpy as np
from scipy.spatial.transform import Rotation


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
        self.register_buffer('line_relevance', torch.tensor(line_relevance, dtype = torch.float32))

        self.register_buffer('_error_buffer', torch.empty(n_points +n_lines,2, dtype = torch.float32))

        lines = torch.tensor(lines_2d, dtype = torch.float32)
        x1, y1, x2, y2 = lines[:,0], lines[:,1], lines[:,2], lines[:,3]

        dx = x2 - x1
        dy = y2 - y1

        self.register_buffer('obs_lines_dx', dx)
        self.register_buffer('obs_lines_dy', dy)
        self.register_buffer('lines_dist_norm', line_relevance/torch.sqrt(dx**2 + dy**2).clamp(min=1e-6))
        self.register_buffer('obs_lines_c', x2*y1 - y2*x1)



    def forward(self):
        #reproject points
        fx, fy, cx, cy = self.intrinsics

        cp = self.cam_t_base_se3.Act(self.points_3d_combined)

        n = cp[..., :2]/cp[..., [2]]

        u = n[..., 0:1]*fx+cx
        v = n[..., 1:2]*fy+cy
        proj_points = torch.cat([u, v], dim=-1)

        # Point errors
        self._error_buffer[:self.n_points] = (proj_points[:self.n_points]-self.observed_points_2d) * (1-self.line_relevance)
    
        # Line errors
        proj_lines_endpoints = proj_points[self.n_points:].reshape(self.n_lines,2, 2)
        px = proj_lines_endpoints[..., 0]
        py = proj_lines_endpoints[..., 1]
        line_distances = (px * self.obs_lines_dy[:,None] - py * self.obs_lines_dx[:,None] + self.obs_lines_c[:,None]) * self.lines_dist_norm[:, None]
        self._error_buffer[self.n_points:] = line_distances

        return self._error_buffer



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
    

def optimize_pnpl(
        initial_cam_t_base:np.ndarray,
        points_3d:np.ndarray,
        points_2d:np.ndarray,
        lines_2d:np.ndarray,
        lines_3d:np.ndarray,
        intrinsic_cam_mat:np.ndarray,
        steps:int = 20,
        reject:int = 30,
        line_relevance:float = 1.0
)->np.ndarray:
    """
    :param initial_cam_t_base: 4x4 homogeneous matrix of the initial camera position
    :param points_3d: Nx3 array of points in the base_frame
    :param points_2d: Nx2 array of observed points in image coordinates (p3d_i corresponds to p2d_i) (with [wi, hi] indexing)
    :param lines_2d: Mx4 array of lines in the format [x1, y1, x2, y2] (l2d_i corresponds to l3d_i)
    :param lines_3d: Mx6 array of lines in the format [x1, y1, z1, x2, y2, z2]
    :param intrinsic_cam_mat: 3x3 Intrinsic camera matrix
    :param steps: The max number of steps the optimizer does
    :param reject: rejection parameter of the LM
    :param line_relevance: multiplier before the line relevance (point relevance = 1-line_relevance)
    """

    assert points_3d.shape[0] == points_2d.shape[0]
    assert points_2d.ndim == 2 and points_2d.shape[-1] == 2
    assert points_3d.ndim == 2 and points_3d.shape[-1] == 3


    assert lines_2d.shape[0] == lines_3d.shape[0]
    assert lines_2d.ndim == 2 and lines_2d.shape[-1] == 4
    assert lines_3d.ndim == 2 and lines_3d.shape[-1] == 6

    assert intrinsic_cam_mat.shape == (3,3)

    assert steps > 0

    assert 0 <= line_relevance <= 1

    model = Reproj(
        cam_intrinsic= intrinsic_cam_mat,
        cam_t_base_quat_vec=hom_to_quat_vec(initial_cam_t_base),
        points_3d=points_3d,
        lines_3d=lines_3d,
        points_2d=points_2d,
        lines_2d=lines_2d,
        line_relevance=line_relevance
    )

    inp = {}

    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5)
    opt = LM(model, solver=Cholesky(), strategy=strategy, reject=reject, sparse=False)
    losses = []
    for step in range(steps):
        loss = opt.step(inp)
        losses.append(loss)
        if len(losses) > 2 and abs(losses[-1]-losses[-2]) < 1e-6:
            break

    final_cam_t_base = model.cam_t_base_se3.matrix().detach().cpu().numpy()
    return final_cam_t_base