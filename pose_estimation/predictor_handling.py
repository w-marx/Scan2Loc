import numpy as np
from dataclasses import dataclass
import sys, os, time, cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_environment import RobotEnvironment
from headset_data import HeadsetData
from shared_utilities import *
from tqdm import tqdm
from time_tracker import TimeTracker
from abc import ABC, abstractmethod

class PosePredictor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def est_base_t_cam2(self,cam2_bgr_image: np.ndarray,time_tracker:TimeTracker) -> np.ndarray | None:
        """
        Predicts the homogenous transformation base_t_cam2
        """
        return None
    
    @abstractmethod
    def update_pose(self,cam2_bgr_image: np.ndarray, old_pose:np.ndarray, time_tracker:TimeTracker) -> np.ndarray | None:
        """
        
        """
        return None
    

@dataclass(frozen=True, kw_only=True)
class RansacPoseEstimationConfig:
    """
    Sets the parameters for an RANSAAC 3d pose estimation.
    """
    min_number_inlier_afterwards:int = 6
    itterations:int = 500
    reprojection_error:float = 5.0
    confidence:float = 0.9
    method = cv2.SOLVEPNP_EPNP

    def __post_init__(self):
        assert 0 < self.min_number_inlier_afterwards
        assert 0 < self.itterations
        assert 0 <= self.reprojection_error
        assert 0 <= self.confidence <= 1.0

pose_estimation_ransaac_config_10ms = RansacPoseEstimationConfig(
    min_number_inlier_afterwards = 6,
    itterations = 500,
    reprojection_error = 5.0,
    confidence = 0.9
)

pose_estimation_ransaac_config_precise = RansacPoseEstimationConfig(
    min_number_inlier_afterwards = 6,
    itterations = 10000,
    reprojection_error = 5.0,
    confidence = 0.99
)

def estimate_point_pose_ransac(
        img_points:np.ndarray, 
        world_points:np.ndarray,
        intrinsic_matrix:np.ndarray, 
        config:RansacPoseEstimationConfig
    )->tuple[np.ndarray, np.ndarray]|None:
    """
    Solves for the cam_t_world position using ransac
    :param img_points: Nx2 array of 2d points [[x1, y1], ...] wher pi in img_points corresponds to pi in world_points
    :param world_points: Nx3 array of 3d points [[x1, y1, z1], ...]
    :param intrinsic_matrix: 3x3 intrinsic matrix
    :param config: The RANSAAC configuration to use
    :return None if optimisation fails, else tuple[cam_t_base, inlier_indices] (cam_t_base is 4x4 hom)
    """

    number_points = img_points.shape[0]
    assert world_points.shape[0] == number_points, f"cant solve: {number_points} & {world_points.shape[0]} points"
    assert img_points.ndim == 2 and img_points.shape[-1] == 2, f"wrong 2d pc shape: {img_points.shape}"
    assert world_points.ndim == 2 and world_points.shape[-1] == 3, f"wrong 2d pc shape: {world_points.shape}"


    if world_points.shape[0] < min(5, config.min_number_inlier_afterwards):
        return None

    success, r_img_t_obj, t_img_t_obj, inliers = cv2.solvePnPRansac(
        world_points, img_points, intrinsic_matrix, None,
        iterationsCount = config.itterations,
        reprojectionError=config.reprojection_error,
        confidence = config.confidence,
        flags = config.method
    )
        
    if not success or len(inliers) < config.min_number_inlier_afterwards:
        return None
    
    cam_t_world= np.eye(4)
    cam_t_world[:3, :3] = cv2.Rodrigues(r_img_t_obj)[0]
    cam_t_world[:3, 3] = t_img_t_obj.flatten()
    return cam_t_world, inliers.flatten()


class OnePredictorRecordingGrader:
    def __init__(self, 
                predictor:PosePredictor, 
                headset_rec:HeadsetData,
                prediction_time_tracker:TimeTracker = TimeTracker(),
                subcomponent_time_tracker:TimeTracker = TimeTracker()
                ) -> None:
        self._predictor = predictor
        self._headset_rec = headset_rec

        prediction_time_tracker.reset_elapsed_time()
        pred_b_t_h_s = []
        for  image in tqdm(headset_rec.headset_bgr_image_s):
            pred_b_t_h_s.append(predictor.est_base_t_cam2(image, subcomponent_time_tracker))
            prediction_time_tracker.add_time_stamp("Single frame from scratch prediction")

        self.pred_b_t_h_s = pred_b_t_h_s

        self.b_t_h_s_np = np.array([b_t_h for b_t_h in headset_rec.robot_base_t_headset_s if b_t_h is not None])
        self.pred_b_t_h_s_np = np.array([b_t_h for b_t_h in pred_b_t_h_s if b_t_h is not None])

    def __str__(self):
        return f"{self._predictor} on {self._headset_rec.name}"
    
    def translational_errors(self) -> list[float]:
        return [
            np.linalg.norm(b_t_h[:3,3]-pred_b_t_h[:3,3])
            for b_t_h, pred_b_t_h in zip(self._headset_rec.robot_base_t_headset_s, self.pred_b_t_h_s) if b_t_h is not None and pred_b_t_h is not None
        ]
    
    def rotational_errors(self) -> list[float]:
        return [
            calc_rotational_difference(pred_b_t_h, b_t_h) 
            for pred_b_t_h, b_t_h in zip(self.pred_b_t_h_s_np, self.b_t_h_s_np)
        ]
    
    def median_translational_error(self) -> float | None:
        return np.median(np.array(self.translational_errors())) if len(self.translational_errors()) > 0 else None

    def median_rotational_error(self) -> float | None:
        return np.median(np.array(self.rotational_errors())) if len(self.rotational_errors()) > 0 else None
    
    def avg_translational_error(self) -> float | None:
        return np.mean(np.array(self.translational_errors())) if len(self.translational_errors()) > 0 else None

    def avg_rotational_error(self) -> float | None:
        return np.mean(np.array(self.rotational_errors())) if len(self.rotational_errors()) > 0 else None
    
    def sucess_ratio(self):
        return self.pred_b_t_h_s_np.shape[0]/len(self.pred_b_t_h_s)
    
    
