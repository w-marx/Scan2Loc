import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numbers import Real

from shared.se3_utilities import translational_difference, rotational_difference, ate_rmse, rte_rotational_errors_rmse, rte_translational_errors_rmse
from shared.image_camera_manipulation import create_3d_camera

from ..data_interfaces.scanned_3d_environment import Scanned3dEnvironment
from ..data_interfaces.headset_data import HeadsetData
from ..utilities.time_tracker import TimeTracker

from ..utilities.slam2mp4 import VideoGenerator, InfoCard
from .gripping_error import sample_pixel_neighborhood, FastGrippingError
from .pose_predictor import PosePredictor


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
                 gripping_error:FastGrippingError | None = None,
                 use_tqdm:bool = True
            ):
        """
        Uses the predictor to run predictions on the dataset and gather metrics.
        :param predictor: The predictor that will do the predictions
        :param headset_data: The headset data that provides the prediction frames & maybe labels
        :param number_retry: Max number of retries given to est_base_t_cam
        :param gripping_error: 
        """
        assert isinstance(predictor, PosePredictor)
        assert isinstance(headset_data, HeadsetData)
        assert isinstance(number_retry, int) and number_retry > 0


        self._headset_data = headset_data

        self._est_base_t_cam_time_tracker = TimeTracker()
        self._per_frame_prediction_time_tracker = TimeTracker()

        self._predictions_whole_time_tracker = TimeTracker()


        self.predicted_base_t_headset_s = []

        for i in (tqdm(range(headset_data.n_frames)) if use_tqdm else range(headset_data.n_frames)):
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
        self.comparable_poses = [
            (i, predicted, label)
            for i, (predicted, label) in enumerate(zip(self.predicted_base_t_headset_s, headset_data.robot_base_t_headset_s))
            if predicted is not None and label is not None
        ]

        self.number_error_computable_poses = len(self.comparable_poses)

        self.timed_translational_errors = [
            (i, translational_difference(m1, m2))
            for i, m1, m2 in self.comparable_poses
        ]

        self.timed_rotational_errors = [
            (i, rotational_difference(m1, m2))
            for i, m1, m2 in self.comparable_poses
        ]

        h, w = headset_data.bgr_image_s[0].shape[:2]
        middle_pixels = sample_pixel_neighborhood((w//2, h//2), size=5)

        self.timed_gripping_errors = []

        if gripping_error is not None:
            for i, m1, m2 in self.comparable_poses:
                e = gripping_error.calculate_gripping_differences_4_pixels(
                    base_t_cam_s= np.array([m1, m2]),
                    pixels_batch= np.array([middle_pixels, middle_pixels]),
                    distance_type='median',
                )[0,1]
                if np.isfinite(e):
                    self.timed_gripping_errors.append((i, e))

        self.avg_gripping_error = np.mean([e for _, e in self.timed_gripping_errors]) if len(self.timed_gripping_errors) > 0 else None
        self.median_gripping_error = np.median([e for _, e in self.timed_gripping_errors]) if len(self.timed_gripping_errors) > 0 else None

        self.translational_errors = [e for _, e in self.timed_translational_errors]
        self.rotational_errors = [e for _, e in self.timed_rotational_errors]

        self.avg_translational_error = np.mean(self.translational_errors) if len(self.translational_errors) > 0 else None
        self.avg_rotational_error = np.mean(self.rotational_errors) if len(self.rotational_errors) > 0 else None

        self.median_translational_error = np.median(self.translational_errors) if len(self.translational_errors) > 0 else None
        self.median_rotational_error = np.median(self.rotational_errors) if len(self.rotational_errors) > 0 else None


        if len(self.comparable_poses) > 0:
            timestamps_sync = [i for i, _, _ in self.comparable_poses]
            predicted_sync = np.asarray([predicted for _, predicted, _ in self.comparable_poses])
            actual_sync = np.asarray([actual for _, _, actual in self.comparable_poses])
        else:
            timestamps_sync = []
            predicted_sync = np.empty((0,4,4))
            actual_sync = np.empty((0,4,4))

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

        # Extract and match metrics:
        self.avg_number_of_tries = None
        self.avg_number_of_inliers = None
        if predictor.extract_and_match_wrapper is not None:
            self.avg_number_of_tries = predictor.extract_and_match_wrapper.get_avg_number_of_tries()
            self.avg_number_of_inliers = predictor.extract_and_match_wrapper.get_avg_number_of_inliers()

    def get_prediction_times(self)->tuple[float | None, list[tuple[str, float]]]:
        summed_time = self._per_frame_prediction_time_tracker.get_timestamp_name_avg_time("predicted 1 frame success")
        subcomponent_times = self._est_base_t_cam_time_tracker.return_averaged_times()
        return (summed_time, subcomponent_times)


    def print_summary(self)->None:
        print(f"Success rate: {self.success_ratio} for {self.number_attempted_predictions} predictions")

        print(f"Avg. time per successful prediction: {format_optional(self.time_per_successful_prediction, fmt=".1f", factor=1000)}ms")
        print(f"est_base_t_cam subcomponent times:\n")
        self._est_base_t_cam_time_tracker.print_report()

        print(f"\n\nAvg. error: {format_optional(self.avg_translational_error, factor=1000)} mm and {format_optional(self.avg_rotational_error, factor=180/np.pi)}°")
        print(f"Median. error: {format_optional(self.median_translational_error, factor=1000)} mm and {format_optional(self.median_rotational_error, factor=180/np.pi)}°")
        print(f"ATE RMSE: {self.ate_translation_rmse * 1000:.1f} mm and {np.rad2deg(self.ate_rot_rmse):.1f}°")
        print(f"RTE RMSE: {self.rte_translation_rmse * 1000:.1f} mm and {np.rad2deg(self.rte_rotational_rmse):.1f}°")
        print(f"avg gripping error: {format_optional(self.avg_gripping_error, fmt=".1f", factor=1000)} mm")
        print(f"median gripping error: {format_optional(self.median_gripping_error, fmt=".1f", factor=1000)} mm")
    

    def visualize_predictions(self, robot_env:Scanned3dEnvironment|None = None, show_label:bool = False, vis_robot_cams:bool = False)->None:
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
            to_vis = robot_env.visualize_3d_data(visualize=False, visualize_robot_cameras=vis_robot_cams)

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
