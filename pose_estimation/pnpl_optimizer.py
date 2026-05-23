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
                xyz_points:np.ndarray,
                lines_3d:np.ndarray,
                points_2d:np.ndarray,
                lines_2d:np.ndarray,
                line_relevance:float
                ):
        super().__init__()
        """
        :param cam_intrinsic: 3x3 intrinsic camera matrix (numpy array)
        :param cam_t_base_quat_vec: length 7 array of the camera pose
        :param xyz_points: Nx3 point_cloud of xyz-points in the base-frame
        """
        self.cam_t_base_se3 = pp.Parameter(pp.SE3(torch.tensor(cam_t_base_quat_vec, dtype = torch.float32)))
        self.register_buffer('cam_intrinsic', torch.tensor(cam_intrinsic, dtype = torch.float32))
        self.register_buffer('xyz_points', torch.tensor(xyz_points, dtype = torch.float32))
        self.register_buffer('lines_3d', torch.tensor(lines_3d, dtype = torch.float32))

        self.register_buffer('observed_points_2d', torch.tensor(points_2d, dtype = torch.float32))
        self.register_buffer('observed_lines_2d', torch.tensor(lines_2d, dtype = torch.float32))
        self.register_buffer('line_relevance', torch.tensor(line_relevance, dtype = torch.float32))


    def forward(self):
        # Point error
        proj_points, valid_z_mask = Reproj.reproject_points(
            points3d=self.xyz_points,
            cam_intrinsic_mtx=self.cam_intrinsic,
            cam_t_base_se3=self.cam_t_base_se3,
            filter_negative_z=True
        )
        point_error = proj_points-self.observed_points_2d[valid_z_mask]

        if self.observed_lines_2d.shape[0] == 0:
            return point_error

        # Line error
        proj_line_end_points, _ = Reproj.reproject_points(
            points3d=self.lines_3d.reshape(-1,3),
            cam_intrinsic_mtx=self.cam_intrinsic,
            cam_t_base_se3=self.cam_t_base_se3,
            filter_negative_z= False
        )

        line_distances = Reproj.distance_lines_points(
            lines=self.observed_lines_2d,
            points=proj_line_end_points.reshape(self.observed_lines_2d.shape[0],2, 2)
        )

        return torch.cat([(1-self.line_relevance)*point_error, self.line_relevance*line_distances], dim = 0)
    
    @staticmethod
    def reproject_points(
            points3d, 
            cam_intrinsic_mtx, 
            cam_t_base_se3, 
            filter_negative_z:bool = False
        ):
        cp = cam_t_base_se3.Act(points3d)


        valid_z = torch.ones(points3d.shape[0], dtype=torch.bool, device=points3d.device)
        if filter_negative_z:
            valid_z = cp[..., 2] > 1e-4
            cp = cp[valid_z]

        n = cp[..., :2]/cp[..., [2]]
        
        fx, fy = cam_intrinsic_mtx[0, 0], cam_intrinsic_mtx[1, 1]
        cx, cy = cam_intrinsic_mtx[0, 2], cam_intrinsic_mtx[1, 2]

        u = n[..., 0:1]*fx+cx
        v = n[..., 1:2]*fy+cy

        proj_points = torch.cat([u, v], dim=-1)
        return proj_points, valid_z
    
    @staticmethod
    def distance_line_points(line:torch.tensor, points:torch.tensor):
        """
        :param line: A 2D line in the form: [x1, y1, x2, y2]
        :param points: Nx2 array of 2d points
        :return an array of length 1 of the signed distances between the line and the points
        """
        x1, y1, x2, y2 = line
        dx, dy = x2-x1, y2-y1
        line_points_dist = torch.sqrt(dx**2 + dy**2).clamp(min= 1e-6)
        c = x2*y1-y2*x1

        point_distances = (points[:, 0] * dy - points[:, 1]*dx + c)/line_points_dist

        return point_distances
    
    @staticmethod
    def distance_lines_points(lines:torch.tensor, points:torch.tensor):
        """
        :param lines: N 2d lines in the form: Nx4
        :param points: NxMx2 array of 2d points
        :return an NxM array of the signed point line distances
        """
        assert lines.ndim == 2 and lines.shape[-1] == 4
        assert points.ndim == 3 and points.shape[-1] == 2
        assert lines.shape[0] == points.shape[0], f"lines: {lines.shape}, points: {points.shape}"


        x1, y1, x2, y2 = lines[:,0], lines[:,1], lines[:,2], lines[:,3]

        dx = x2 - x1
        dy = y2 - y1

        denom = torch.sqrt(dx**2 + dy**2).clamp(min=1e-6)

        c = x2*y1 - y2*x1

        px = points[..., 0]
        py = points[..., 1]

        point_distances = (px * dy[:,None] - py * dx[:,None] + c[:,None]) / denom[:,None]
        return point_distances



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
    :param reject: ?
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
        xyz_points=points_3d,
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