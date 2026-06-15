import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numbers import Number, Real

from shared.assertion_helpers import assert_homogeneous_mat, assert_intrinsic_mat
from shared.se3_utilities import translational_difference, rotational_difference, ate_rmse, rte_rotational_errors_rmse, rte_translational_errors_rmse
from shared.image_camera_manipulation import create_3d_camera

from .robot_environment import RobotEnvironment
from .headset_data import HeadsetData
from .geometric_utilities.time_tracker import TimeTracker
from .geometric_utilities.slam2mp4 import VideoGenerator, FeatureDrawing, InfoCard


from .predictor_handling import PosePredictor


def calculate_reprojection_metrics(
        cam_t_base1:np.ndarray, 
        cam_t_base2:np.ndarray, 
        shared_intrinsic_mat:np.ndarray, 
        points_3d:np.ndarray
    )->tuple[float, float] | None:

    assert assert_homogeneous_mat(cam_t_base1, size = 4) and assert_homogeneous_mat(cam_t_base2, size = 4)
    assert assert_intrinsic_mat(shared_intrinsic_mat)
    assert points_3d.ndim == 2 and points_3d.shape[-1] == 3

    if points_3d.shape[0] == 0:
        return None

    points_hom = np.column_stack([points_3d, np.ones(points_3d.shape[0])])

    P1 = shared_intrinsic_mat @ cam_t_base1[:3, :]
    P2 = shared_intrinsic_mat @ cam_t_base2[:3, :]

    cam_points1 = (P1 @ points_hom.T).T
    cam_points2 = (P2 @ points_hom.T).T

    projectable_mask = (cam_points1[:, 2] > 1e-10) & (cam_points2[:, 2] > 1e-10)
    cam_points1 = cam_points1[projectable_mask[:]]
    cam_points2 = cam_points2[projectable_mask[:]]

    projected1 = cam_points1[:, :2]/cam_points1[:, 2:3]
    projected2 = cam_points2[:, :2]/cam_points2[:, 2:3]

    errors = np.linalg.norm(projected1-projected2, axis=-1)

    avg_error = np.mean(errors)
    median_error = np.median(errors)

    return avg_error, median_error


def format_optional(value:Real | float | int | np.floating | None, fmt=".1f", default = "N/A", factor:float = 1.0):
    if value is None:
        return default
    return f"{(value*factor):{fmt}}"


class PredictionOnDataset:
    def __init__(self,
                 predictor:PosePredictor,
                 headset_data:HeadsetData,
                 number_retry:int = 1,
                 vid_gen:VideoGenerator | None = None,
                 video_save_location:str = "test.mp4",
                 point_cloud:np.ndarray | None = None
            ):
        """
        Uses the predictor to run predictions on the dataset and gather metrics.
        :param predictor: The predictor that will do the predictions
        :param headset_data: The headset data that provides the prediction frames & maybe labels
        :param number_retry: Max number of retries given to est_base_t_cam
        """
        assert isinstance(predictor, PosePredictor)
        assert isinstance(headset_data, HeadsetData)
        assert isinstance(number_retry, int) and number_retry > 0


        self._headset_data = headset_data

        self._est_base_t_cam_time_tracker = TimeTracker()
        self._per_frame_prediction_time_tracker = TimeTracker()

        self._predictions_whole_time_tracker = TimeTracker()


        self.predicted_base_t_headset_s = []

        for i in tqdm(range(headset_data.n_frames)):
            self._per_frame_prediction_time_tracker.reset_elapsed_time()
            headset_image = headset_data.bgr_image_s[i]

            if vid_gen is not None:
                vid_gen.start_new_frame(headset_image)

            self._predictions_whole_time_tracker.reset_elapsed_time()
            est_base_t_cam = predictor.est_base_t_cam2(
                cam2_bgr_image=headset_image,
                number_retry=number_retry,
                time_tracker=self._est_base_t_cam_time_tracker,
                fd=vid_gen.current_feature_drawer if vid_gen is not None else None
            )
            self._predictions_whole_time_tracker.add_time_stamp(f"1 est_base_t_cam call {"fail" if est_base_t_cam is None else "success"}")
            self._per_frame_prediction_time_tracker.add_time_stamp(f"predicted 1 frame {"fail" if est_base_t_cam is None else "success"}")
            self.predicted_base_t_headset_s.append(est_base_t_cam)

            if vid_gen is not None:
                vid_gen.annotate_frame(
                    info_card=InfoCard(
                        predicted_base_t_cam=est_base_t_cam,
                        actual_base_t_cam=headset_data.robot_base_t_headset_s[i]
                        )
                    )
                vid_gen.end_current_frame()
        
        if vid_gen is not None:
            vid_gen.save_video(location=video_save_location)


        self.predicted_base_t_headset_s_no_none = [
            b_t_h
            for b_t_h in self.predicted_base_t_headset_s if b_t_h is not None
        ]

        # Success metrics
        self.number_attempted_predictions = len(self.predicted_base_t_headset_s)
        self.number_successful_predictions = len(self.predicted_base_t_headset_s_no_none)
        self.success_ratio = self.number_successful_predictions / self.number_attempted_predictions

        # Time metrics
        self.time_per_successful_prediction = self._per_frame_prediction_time_tracker.get_timestamp_name_avg_time("predicted 1 frame success")
        self.time_per_failed_prediction = self._per_frame_prediction_time_tracker.get_timestamp_name_avg_time("predicted 1 frame fail")

        self.avg_time_for_frame_prediction = self._per_frame_prediction_time_tracker.get_timestamp_name_avg_time("predicted 1 frame")

        # Accuracy metrics
        comparable_poses = [
            (i, predicted, label)
            for i, (predicted, label) in enumerate(zip(self.predicted_base_t_headset_s, headset_data.robot_base_t_headset_s))
            if predicted is not None and label is not None
        ]

        self.number_error_computable_poses = len(comparable_poses)

        self.timed_reprojection_errors_avg_med = []
        if point_cloud is not None:
            for i, m1, m2 in comparable_poses:
                avg__med = calculate_reprojection_metrics(
                    cam_t_base1=m1, 
                    cam_t_base2=m2, 
                    shared_intrinsic_mat=headset_data.intrinsic_cam_mtx,
                    points_3d=point_cloud
                )
                if avg__med is not None:
                    self.timed_reprojection_errors_avg_med.append((i, avg__med[0], avg__med[0]))

        self.avg_reprojection_error = None
        self.median_reprojection_error = None
        if len(self.timed_reprojection_errors_avg_med) > 0:
            self.avg_reprojection_error = np.mean([avg for _, avg, _ in self.timed_reprojection_errors_avg_med])
            self.median_reprojection_error = np.median([median for _, _, median in self.timed_reprojection_errors_avg_med])


        self.timed_translational_errors = [
            (i, translational_difference(m1, m2))
            for i, m1, m2 in comparable_poses
        ]

        self.timed_rotational_errors = [
            (i, rotational_difference(m1, m2))
            for i, m1, m2 in comparable_poses
        ]

        self.translational_errors = [e for _, e in self.timed_translational_errors]
        self.rotational_errors = [e for _, e in self.timed_rotational_errors]

        self.avg_translational_error = np.mean(self.translational_errors)
        self.avg_rotational_error = np.mean(self.rotational_errors)

        self.median_translational_error = np.median(self.translational_errors)
        self.median_rotational_error = np.median(self.rotational_errors)

        timestamps_sync = [i for i, _, _ in comparable_poses]
        predicted_sync = np.asarray([predicted for _, predicted, _ in comparable_poses])
        actual_sync = np.asarray([actual for _, _, actual in comparable_poses])

        self.ate_translation_rmse, self.ate_rot_rmse = ate_rmse(
            predicted=predicted_sync,
            actual=actual_sync
        )
        self.rte_translation_errors, self.rte_translation_rmse = rte_translational_errors_rmse(
            timestamps=timestamps_sync,
            predicted=predicted_sync,
            actual=actual_sync
        )
        self.rte_rotational_errors, self.rte_rotational_rmse = rte_rotational_errors_rmse(
            timestamps=timestamps_sync,
            predicted=predicted_sync,
            actual=actual_sync
        )

    def print_summary(self)->None:
        print(f"Success rate: {self.success_ratio} for {self.number_attempted_predictions} predictions")

        print(f"Avg. time per successful prediction: {format_optional(self.time_per_successful_prediction, fmt=".1f", factor=1000)}ms")
        print(f"est_base_t_cam subcomponent times:\n")
        self._est_base_t_cam_time_tracker.print_report()

        print(f"\n\nAvg. error: {self.avg_translational_error*1000:.1f} mm and {np.rad2deg(self.avg_rotational_error):.1f}°")
        print(f"Median. error: {self.median_translational_error*1000:.1f} mm and {np.rad2deg(self.median_rotational_error):.1f}°")
        print(f"ATE RMSE: {self.ate_translation_rmse * 1000:.1f} mm and {np.rad2deg(self.ate_rot_rmse):.1f}°")
        print(f"RTE RMSE: {self.rte_translation_rmse * 1000:.1f} mm and {np.rad2deg(self.rte_rotational_rmse):.1f}°")
        print(f"avg reprojection error: {format_optional(self.avg_reprojection_error, fmt=".1f")} px")
        print(f"median reprojection error: {format_optional(self.median_reprojection_error, fmt=".1f")} px")


    #def get_subcomponent_times_est_base_t_cam_call(self)-> list[tuple[str, float]]:
    #    """
    #    Returns the avg. times and their subcomponents per est_base_t_cam call
    #    :return: The time per call and a list of [subcomponent_name, avg time in seconds] tuples (sorted by time descending)
    #    """
    #    complete_time = self._predictions_whole_time_tracker.get_timestamp_name_avg_time("1 est_base_t_cam call")
    #    sub_times = self._est_base_t_cam_time_tracker.return_averaged_times()
    #    return complete_time, sub_times
    

    def visualize_predictions(self, robot_env:RobotEnvironment|None = None, show_label:bool = False)->None:
        """
        Visualizes the predictions made by the predictor using open3d
        :param robot_env: RobotEnvironment or None, if not None will be added to the plot
        :param show_label: whether to show the label camera frames or not
        """
        colors = plt.cm.jet(np.linspace(0, 1, self._headset_data.n_frames))[:, :3]

        to_vis = []
        if robot_env is None:
            base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            to_vis.append(base_frame)
        else:
            to_vis = robot_env.visualize_3d_data(visualize=False)

        for i, (predicted_b_t_h, b_t_h) in enumerate(zip(self.predicted_base_t_headset_s, self._headset_data.robot_base_t_headset_s)):
            if b_t_h is not None and show_label:
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005)
                cam_frame.transform(b_t_h)
                to_vis.append(cam_frame)
                camera_line_set = create_3d_camera(
                    base_t_camera=b_t_h,
                    intrinsics=self._headset_data.intrinsic_cam_mtx,
                    hxw_img=self._headset_data.bgr_image_s[i],
                    scale=0.01
                )
                camera_line_set.paint_uniform_color(colors[i])
                to_vis.append(camera_line_set)

            if predicted_b_t_h is not None:
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005)
                cam_frame.transform(predicted_b_t_h)
                to_vis.append(cam_frame)
                camera_line_set = create_3d_camera(
                    base_t_camera=predicted_b_t_h,
                    intrinsics=self._headset_data.intrinsic_cam_mtx,
                    hxw_img=self._headset_data.bgr_image_s[i],
                    scale=0.01
                )
                camera_line_set.paint_uniform_color(colors[i])
                to_vis.append(camera_line_set)

            if b_t_h is not None and predicted_b_t_h is not None and show_label:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector([b_t_h[:3,3], predicted_b_t_h[:3,3]])
                line_set.lines = o3d.utility.Vector2iVector([[0,1]])
                line_set.paint_uniform_color(colors[i])
                to_vis.append(line_set)

        if len(self.predicted_base_t_headset_s_no_none) > 0:
            pred_line_set = o3d.geometry.LineSet()
            pred_line_set.points = o3d.utility.Vector3dVector(np.array(self.predicted_base_t_headset_s_no_none)[:,:3,3])
            pred_line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(self.predicted_base_t_headset_s_no_none)-1)])
            pred_line_set.paint_uniform_color([1,0,0])
            to_vis.append(pred_line_set)

        obs_line_set = o3d.geometry.LineSet()
        points = np.array([m for m in self._headset_data.robot_base_t_headset_s if m is not None])[:, :3, 3]
        obs_line_set.points = o3d.utility.Vector3dVector(points)
        obs_line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(points)-1)])
        obs_line_set.paint_uniform_color([0,1,0])
        to_vis.append(obs_line_set)

        o3d.visualization.draw_geometries(to_vis, f"Headset Predictions visualization")
