import numpy as np
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_2_prediction import PredictionData, create_3d_camera
from gathering_2_preprocessing import calc_rotational_difference, compute_pose_pseudo_median
from tqdm import tqdm
from time_tracker import TimeTracker

class PosePredictor:
    def __init__(self):
        pass

    def est_base_t_cam2(self,
                        cam1_bgr_image:np.ndarray,
                        base_xyz_image:np.ndarray,
                        cam2_bgr_image: np.ndarray,
                        point_cloud:np.ndarray,
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
                        point_cloud:np.ndarray,
                        time_tracker:TimeTracker
                        ) -> list[np.ndarray | None]:
        """
        Predicts the homogenous transformations base_t_cam2
        :param cam1_bgr_image_s: An NxHxWx3-uint8 numpy array of the images from the POV of cam1
        :param cam1_base_xyz_image_s: An NxHxWx3-float numpy array of the world xyz-coordinates of the pixels in cam1_bgr_images
        :param cam2_bgr_image_s: An NxHxWx3-uint8 numpy array of the images from the POV of cam2
        :param point_cloud: A Nx3-float point-cloud of the sorroundings in base-frame coordinates
        """
        num_datapoints = cam1_bgr_image_s.shape[0]
        assert num_datapoints > 0
        assert cam1_bgr_image_s.ndim == 4 and cam1_bgr_image_s.shape[-1] == 3
        assert cam1_base_xyz_image_s.shape == cam1_bgr_image_s.shape
        assert cam2_bgr_image_s.shape[0] == num_datapoints and cam2_bgr_image_s.ndim == 4 and cam2_bgr_image_s.shape[-1] == 3
        assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 3

        predicted_base_t_cam2_s = []
        print("predicting base_t_cam2_s batched")
        for cam1_img, xyz_img, cam2_img in tqdm(zip(cam1_bgr_image_s, cam1_base_xyz_image_s, cam2_bgr_image_s), total=num_datapoints):
            predicted_base_t_cam2_s.append(self.est_base_t_cam2(
                cam1_bgr_image=cam1_img,
                base_xyz_image=xyz_img,
                cam2_bgr_image=cam2_img,
                point_cloud=point_cloud,
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
            point_cloud=data.point_cloud,
            time_tracker=time_tracker
        )


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
        self._avg_time_per_successful_prediction = (end_time-start_time)/len(self._base_t_headsets_no_none)


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
    
    
