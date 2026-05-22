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
                lines_3d:np.ndarray
                ):
        super().__init__()
        """
        :param cam_intrinsic: 3x3 intrinsic camera matrix (numpy array)
        :param cam_t_base_quat_vec: length 7 array of the camera pose
        :param xyz_points: Nx3 point_cloud of xyz-points in the base-frame
        """
        self.cam_t_base_se3 = pp.Parameter(pp.SE3(torch.tensor(cam_t_base_quat_vec, dtype = torch.float64)))
        self.register_buffer('cam_intrinsic', torch.tensor(cam_intrinsic))
        self.register_buffer('xyz_points', torch.tensor(xyz_points))
        self.register_buffer('lines_3d', torch.tensor(lines_3d))


    def forward(self, observed_points_2d, observed_lines_2d, line_relevance):
        # Point error
        proj_points, valid_z_mask = Reproj.reproject_points(
            points3d=self.xyz_points,
            cam_intrinsic_mtx=self.cam_intrinsic,
            cam_t_base_se3=self.cam_t_base_se3,
            filter_negative_z=True
        )
        point_error = proj_points-observed_points_2d[valid_z_mask]

        if observed_lines_2d.shape[0] == 0:
            print(f"used points instead")
            return point_error

        # Line error
        proj_line_end_points, _ = Reproj.reproject_points(
            points3d=self.lines_3d.reshape(-1,3),
            cam_intrinsic_mtx=self.cam_intrinsic,
            cam_t_base_se3=self.cam_t_base_se3,
            filter_negative_z= False
        )
        proj_line_end_points = proj_line_end_points.reshape(-1,4)

        
        line_distances = torch.stack([
            Reproj.distance_line_points(line, points.reshape(-1,2))
            for line, points in zip(observed_lines_2d, proj_line_end_points)
        ])

        return torch.cat([(1-line_relevance)*point_error, line_relevance*line_distances], dim = 0)
    
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

        n = cp[..., :2] / cp[..., [2]]
        
        fx, fy = cam_intrinsic_mtx[0, 0], cam_intrinsic_mtx[1, 1]
        cx, cy = cam_intrinsic_mtx[0, 2], cam_intrinsic_mtx[1, 2]

        u = fx * n[..., 0:1] + cx
        v = fy * n[..., 1:2] + cy

        proj_points = torch.cat([u, v], dim=-1)
        return proj_points, valid_z
    
    @staticmethod
    def distance_line_points(line:torch.tensor, points:torch.tensor):
        """
        :param line: A 2D line in the form: [x1, y1, x2, y2]
        :param points: Nx2 array of 2d points
        :return an array of length 1 of the distances between the line and the points
        """
        x1, y1, x2, y2 = line
        dx, dy = x2-x1, y2-y1
        line_points_dist = torch.sqrt(dx**2 + dy**2).clamp(min= 1e-6)
        c = x2*y1-y2*x1


        point_distances = (points[:, 0] * dy - points[:, 1]*dx + c)/line_points_dist

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
        lines_3d=lines_3d
    )

    inp = {
        "observed_points_2d": torch.tensor(points_2d, dtype = torch.float64),
        "observed_lines_2d": torch.tensor(lines_2d, dtype = torch.float64),
        "line_relevance": line_relevance
    }

    strategy = pp.optim.strategy.TrustRegion(up=2.0, down=0.5)
    opt = LM(model, solver=Cholesky(), strategy=strategy, reject=reject, sparse=False)

    for step in range(steps):
        loss = opt.step(inp)
        #print(f"Iteration {step:02d}, loss: {loss.item()}")

    final_cam_t_base = model.cam_t_base_se3.matrix().detach().cpu().numpy()
    return final_cam_t_base