import numpy as np
from dataclasses import dataclass
import sys, os, time, cv2
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_2_prediction import PredictionData
from shared_utilities import *
from tqdm import tqdm
from time_tracker import TimeTracker

class PosePredictor:
    def __init__(self):
        pass

    def est_base_t_cam2(self,
                        cam1_bgr_image:np.ndarray,
                        base_xyz_image:np.ndarray,
                        cam2_bgr_image: np.ndarray,
                        time_tracker:TimeTracker
                        ) -> np.ndarray | None:
        """
        Predicts the homogenous transformation base_t_cam2
        """
        raise NotImplementedError
    
    def est_base_t_cam2_s(self,
                        cam1_bgr_image_s:np.ndarray,
                        cam1_base_xyz_image_s:np.ndarray,
                        cam2_bgr_image_s: np.ndarray,
                        time_tracker:TimeTracker
                        ) -> list[np.ndarray | None]:
        """
        Predicts the homogenous transformations base_t_cam2
        :param cam1_bgr_image_s: An NxHxWx3-uint8 numpy array of the images from the POV of cam1
        :param cam1_base_xyz_image_s: An NxHxWx3-float numpy array of the world xyz-coordinates of the pixels in cam1_bgr_images
        :param cam2_bgr_image_s: An NxHxWx3-uint8 numpy array of the images from the POV of cam2
        """
        num_datapoints = cam1_bgr_image_s.shape[0]
        assert num_datapoints > 0
        assert cam1_bgr_image_s.ndim == 4 and cam1_bgr_image_s.shape[-1] == 3
        assert cam1_base_xyz_image_s.shape == cam1_bgr_image_s.shape
        assert cam2_bgr_image_s.shape[0] == num_datapoints and cam2_bgr_image_s.ndim == 4 and cam2_bgr_image_s.shape[-1] == 3

        predicted_base_t_cam2_s = []
        print("predicting base_t_cam2_s batched")
        for cam1_img, xyz_img, cam2_img in tqdm(zip(cam1_bgr_image_s, cam1_base_xyz_image_s, cam2_bgr_image_s), total=num_datapoints):
            predicted_base_t_cam2_s.append(self.est_base_t_cam2(
                cam1_bgr_image=cam1_img,
                base_xyz_image=xyz_img,
                cam2_bgr_image=cam2_img,
                time_tracker=time_tracker
            ))
        return predicted_base_t_cam2_s
    
    def est_robot_base_t_headset_s(self, data:PredictionData, time_tracker:TimeTracker)->list[np.ndarray | None]:
        """
        Returns robot_base_t_robot_headset for all robot images in prediction data
        :param data: The data to be acted upon
        :return List of (4x4 homogeneous matricies or None)
        """
        return self.est_base_t_cam2_s(
            cam2_bgr_image_s=np.tile(data.headset_bgr_image, (data.robot_bgr_images.shape[0], 1, 1, 1)),
            cam1_bgr_image_s=data.robot_bgr_images,
            cam1_base_xyz_image_s=data.robot_xyz_images,
            time_tracker=time_tracker
        )
    

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


class OnePredictorOneDatasetGrader:
    def __init__(self, predictor:PosePredictor, data:PredictionData, time_tracker:TimeTracker = None) -> None:
        self._predictor = predictor
        self._data = data
        start_time = time.perf_counter()
        self._time_tracker = TimeTracker() if time_tracker is None else time_tracker
        self._base_t_headsets = predictor.est_robot_base_t_headset_s(data, self._time_tracker)
        end_time = time.perf_counter()
        self._base_t_headsets_no_none = [x for x in self._base_t_headsets if x is not None]
        self._avg_time_per_started_prediction = (end_time-start_time)/len(self._base_t_headsets)

        if len(self._base_t_headsets_no_none) > 0:
            self._avg_time_per_successful_prediction = (end_time-start_time)/len(self._base_t_headsets_no_none)
        else:
            self._avg_time_per_successful_prediction = np.inf


    def __str__(self):
        return f"{self.predictor} on {self._data.name}"
    
    def translational_errors(self) -> list[float]:
        if self._data.robot_base_t_headset is None:
            return []
        return [np.linalg.norm(prediction[:3,3]-self._data.robot_base_t_headset[:3,3]) for prediction in self._base_t_headsets_no_none]
    
    def rotational_errors(self) -> list[float]:
        if self._data.robot_base_t_headset is None:
            return []
        return [calc_rotational_difference(prediction, self._data.robot_base_t_headset) for prediction in self._base_t_headsets_no_none]
    
    def median_translational_error(self) -> float | None:
        return np.median(self.translational_errors()) if len(self.translational_errors()) > 0 else None
    
    def average_sub_median_translat_error(self)->float | None:
        median_error = self.median_translational_error()
        if median_error is None:
            return None
        return np.mean([e for e in self.translational_errors() if e <= median_error])
    
    def average_sub_median_rotational_error(self)->float | None:
        median_error = self.median_rotational_error()
        if median_error is None:
            return None
        return np.mean([e for e in self.rotational_errors() if e <= median_error])

    def median_rotational_error(self) -> float | None:
        return np.median(self.rotational_errors()) if len(self.rotational_errors()) > 0 else None
    
    def median_base_t_headset(self)->np.ndarray:
        return compute_pose_pseudo_median(self._base_t_headsets_no_none)
    
    def visualize_predictions(self):
        import open3d as o3d

        img_pcd = o3d.geometry.PointCloud()
        img_pcd.points = o3d.utility.Vector3dVector(self._data.robot_xyz_images.reshape(-1,3))

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self._data.point_cloud)
        pcd.paint_uniform_color([0, 0, 0])

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

        to_vis = [img_pcd, pcd, base_frame]

        for b_t_h in self._base_t_headsets_no_none:
            est_headset_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            est_headset_frame.transform(b_t_h)
            to_vis.append(est_headset_frame)

        if self._data.robot_base_t_headset is not None:
            headset_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.3)
            to_vis.append(create_3d_camera(
                base_t_camera=self._data.robot_base_t_headset,
                intrinsics=self._data.headset_intrinsics,
                hxw_img=self._data.headset_bgr_image,
                scale=0.3
            ))
        headset_frame.transform(self._data.robot_base_t_headset)
        to_vis.append(headset_frame)

        o3d.visualization.draw_geometries(to_vis, f"Predictions visualization")

    def average_time_per_started_prediction(self):
        return self._avg_time_per_started_prediction
    
    def average_time_per_successful_prediction(self):
        return self._avg_time_per_successful_prediction
    
    def sucess_ratio(self):
        return len(self._base_t_headsets_no_none)/len(self._base_t_headsets)
    
    def print_prediction_time_tracker(self):
        self._time_tracker.print_report()
    
    
