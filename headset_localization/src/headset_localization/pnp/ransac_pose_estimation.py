import numpy as np
from dataclasses import dataclass
import cv2
from enum import Enum

from shared.se3_utilities import r_t_to_hom

from ..utilities.point_utilities import project_visible_points


class OpenCVPnPSolvers(Enum):    
    SOLVEPNP_ITERATIVE = cv2.SOLVEPNP_ITERATIVE
    SOLVEPNP_EPNP = cv2.SOLVEPNP_EPNP
    SOLVEPNP_P3P = cv2.SOLVEPNP_P3P # exactly 4 points
    SOLVEPNP_AP3P = cv2.SOLVEPNP_AP3P # exactly 4 points
    SOLVEPNP_IPPE = cv2.SOLVEPNP_IPPE # For coplanar
    SOLVEPNP_IPPE_SQUARE = cv2.SOLVEPNP_IPPE_SQUARE # For coplanar & 4 points

    def __str__(self):
        return self.name


@dataclass(frozen=True, kw_only=True)
class RansacPoseEstimationConfig:
    """
    Sets the parameters for an RANSAC 3d pose estimation.
    :param min_number_inlier_afterwards: the minimum number of inlier's after RANSAC
    :param iterations: the number of iterations
    :param reprojection_error: the reprojection error for RANSAC
    :param confidence: the confidence for RANSAC
    :param method: The solving method e.g. cv2.SOLVEPNP_EPNP, cv2.SOLVEPNP_ITERATIVE
    """
    min_number_inlier_afterwards:int = 10
    iterations:int = 500
    reprojection_error:float = 5.0
    confidence:float = 0.9
    method:OpenCVPnPSolvers = OpenCVPnPSolvers.SOLVEPNP_EPNP

    def __post_init__(self):
        assert 0 < self.min_number_inlier_afterwards
        assert 0 < self.iterations
        assert 0 <= self.reprojection_error
        assert 0 <= self.confidence <= 1.0

pose_estimation_ransaac_config_10ms = RansacPoseEstimationConfig(
    min_number_inlier_afterwards = 6,
    iterations = 500,
    reprojection_error = 5.0,
    confidence = 0.9
)

pose_estimation_ransaac_config_precise = RansacPoseEstimationConfig(
    min_number_inlier_afterwards = 6,
    iterations = 10000,
    reprojection_error = 5.0,
    confidence = 0.99
)

pose_estimation_ransaac_config_less_precise = RansacPoseEstimationConfig(
    min_number_inlier_afterwards = 6,
    iterations = 1000,
    reprojection_error = 5.0,
    confidence = 0.99
)


def estimate_point_pose_ransac(
        img_points:np.ndarray, 
        world_points:np.ndarray,
        intrinsic_matrix:np.ndarray, 
        config:RansacPoseEstimationConfig,
    )->tuple[np.ndarray, np.ndarray]|None:
    """
    Solves for the cam_t_world position using ransac
    :param img_points: Nx2 array of 2d points [[x1, y1], ...] wher pi in img_points corresponds to pi in world_points
    :param world_points: Nx3 array of 3d points [[x1, y1, z1], ...]
    :param intrinsic_matrix: 3x3 intrinsic matrix
    :param config: The RANSAC configuration to use
    :param ax: Will draw the proj-points <-> observed points onto the axes
    :return None if optimisation fails, else tuple[cam_t_base, inlier_indices] (cam_t_base is 4x4 hom)
    """
    number_points = img_points.shape[0]
    assert world_points.shape[0] == number_points, f"cant solve: {number_points} & {world_points.shape[0]} points"
    assert img_points.ndim == 2 and img_points.shape[-1] == 2, f"wrong 2d pc shape: {img_points.shape}"
    assert world_points.ndim == 2 and world_points.shape[-1] == 3, f"wrong 2d pc shape: {world_points.shape}"

    if world_points.shape[0] < min(5, config.min_number_inlier_afterwards):
        return None

    success, r_img_t_obj, t_img_t_obj, inliers = cv2.solvePnPRansac(
        objectPoints = world_points.astype(np.float64), 
        imagePoints = img_points.astype(np.float64), 
        cameraMatrix = intrinsic_matrix.astype(np.float64), 
        distCoeffs=np.zeros(5, dtype=np.float64),
        iterationsCount = config.iterations,
        reprojectionError=config.reprojection_error,
        confidence = config.confidence,
        flags = int(config.method.value)
    )
        
    if not success or len(inliers) < config.min_number_inlier_afterwards:
        return None
    
    cam_t_base = r_t_to_hom(cv2.Rodrigues(r_img_t_obj)[0], t_img_t_obj.flatten())

    return cam_t_base, inliers.flatten()