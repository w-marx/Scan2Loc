import sys, os
from typing import Callable

import numpy as np

from hom_pose_utilities import translational_difference, rotational_difference, ate_rmse, rte_rotational_errors_rmse, rte_translational_errors_rmse
from image_camera_manipulation import create_3d_camera

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_environment import RobotEnvironment
from headset_data import HeadsetData
from assertion_helpers import *
from time_tracker import TimeTracker
import open3d as o3d
from predictor_handling import PosePredictor
from dataclasses import dataclass

import matplotlib.pyplot as plt



class PredictionOnDataset:
    def __init__(self,
                 predictor:PosePredictor,
                 headset_data:HeadsetData,
                 number_consecutive_update_pose_calls:int = 0,
                 number_retry:int = 1
                ):
        """
        Uses the predictor to run predictions on the dataset and gather metrics.
        It uses the following scheme:
        - get initial pose using `est_base_t_cam`
        - on the next `number_consecutive_update_pose_calls` frames use `update_pose`
        - if `update_pose` fails use `est_base_t_cam` instead

        :param predictor: The predictor that will do the predictions
        :param headset_data: The headset data that provides the prediction frames & maybe labels
        :param number_consecutive_update_pose_calls: Max number of consecutive update_pose_calls
        :param number_retry: Max number of retries given to est_base_t_cam
        """
        assert isinstance(predictor, PosePredictor)
        assert isinstance(headset_data, HeadsetData)
        assert isinstance(number_consecutive_update_pose_calls, int) and number_consecutive_update_pose_calls >= 0
        assert isinstance(number_retry, int) and number_retry > 0


        self._predictor = predictor
        self._headset_data = headset_data

        self._est_base_t_cam_time_tracker = TimeTracker()
        self._update_pose_time_tracker = TimeTracker()
        self._per_frame_prediction_time_tracker = TimeTracker()

        self._predictions_whole_time_tracker = TimeTracker()


        self.predicted_base_t_headset_s = []

        update_pose_successful_count = 0
        update_pose_failed_count = 0

        i = 0
        c = 0
        while i < headset_data.n_frames:
            headset_image = headset_data.bgr_image_s[i]
            c = c % (number_consecutive_update_pose_calls + 1)
            est_base_t_cam = None

            if c != 0 and self.predicted_base_t_headset_s[-1] is not None:
                self._predictions_whole_time_tracker.reset_elapsed_time()
                est_base_t_cam = predictor.update_pose(
                    cam2_bgr_image=headset_image,
                    rough_base_t_cam2=self.predicted_base_t_headset_s[-1],
                    time_tracker=self._update_pose_time_tracker,
                )
                self._predictions_whole_time_tracker.add_time_stamp("1 update_pose call")
                update_pose_successful_count += 0 if est_base_t_cam is None else 1
                update_pose_failed_count += 1 if est_base_t_cam is None else 0
                c += 1

            if est_base_t_cam is None:
                self._predictions_whole_time_tracker.reset_elapsed_time()
                est_base_t_cam = predictor.est_base_t_cam2(
                    cam2_bgr_image=headset_image,
                    number_retry=number_retry,
                    time_tracker=self._est_base_t_cam_time_tracker,
                )
                self._predictions_whole_time_tracker.add_time_stamp("1 est_base_t_cam call")
                c = 1

            self.predicted_base_t_headset_s.append(est_base_t_cam)
            i += 1
            self._per_frame_prediction_time_tracker.add_time_stamp("predicted 1 frame")

        self.predicted_base_t_headset_s_no_none = [
            b_t_h
            for b_t_h in self.predicted_base_t_headset_s if b_t_h is not None
        ]

        # Success metrics
        self.number_attempted_predictions = len(self.predicted_base_t_headset_s)
        self.number_successful_predictions = len(self.predicted_base_t_headset_s_no_none)
        self.success_ratio = self.number_successful_predictions / self.number_attempted_predictions

        # Time metrics
        self.avg_time_for_frame_prediction = self._per_frame_prediction_time_tracker.get_timestamp_name_avg_time("predicted 1 frame")

        # Accuracy metrics
        comparable_poses = [
            (i, predicted, label)
            for i, (predicted, label) in enumerate(zip(self.predicted_base_t_headset_s, headset_data.robot_base_t_headset_s))
            if predicted is not None and label is not None
        ]

        self.number_error_computable_poses = len(comparable_poses)

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

        print(f"Avg. time per prediction: {self.avg_time_for_frame_prediction*1000}ms")
        print(f"est_base_t_cam subcomponent times:\n")
        self._est_base_t_cam_time_tracker.print_report()

        print(f"Avg. error: {self.avg_translational_error*1000}mm and {np.rad2deg(self.avg_rotational_error)}°")
        print(f"Median. error: {self.median_translational_error*1000}mm and {np.rad2deg(self.median_rotational_error)}°")
        print(f"ATE RMSE: {self.ate_translation_rmse * 1000}mm and {np.rad2deg(self.ate_rot_rmse)}°")
        print(f"RTE RMSE: {self.rte_translation_rmse * 1000}mm and {np.rad2deg(self.rte_rotational_rmse)}°")


    def get_avg_time_per_est_base_t_cam_call(self)->tuple[float, list[tuple[str, float]]]:
        """
        Returns the avg. times and their subcomponents per est_base_t_cam call
        :return: The time per call and a list of [subcomponent_name, avg time in seconds] tuples (sorted by time descending)
        """
        complete_time = self._predictions_whole_time_tracker.get_timestamp_name_avg_time("1 est_base_t_cam call")
        sub_times = self._est_base_t_cam_time_tracker.return_averaged_times()
        return complete_time, sub_times

    def get_avg_time_per_update_pose_call(self)->tuple[float, list[tuple[str, float]]]:
        """
        Returns the avg. times and their subcomponents per update_pose call
        :return: The time per call and a list of [subcomponent_name, avg time in seconds] tuples (sorted by time descending)
        """
        complete_time = self._predictions_whole_time_tracker.get_timestamp_name_avg_time("1 update_pose call")
        sub_times = self._update_pose_time_tracker.return_averaged_times()
        return complete_time, sub_times

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


@dataclass(frozen=True, kw_only=True)
class GradablePosePredictor:
    """
    A dataclass to grade a Pose Predictor including its runtime
    :param creator: A function that takes an Robot environment and a TimeTracker for the init and returns the PosePredictor
    :param name: The identifying name of the PosePredictor (may appear on plots)
    :param number_retries: The number of times to retries passed to est_base_t_cam
    :param max_number_consecutive_update_pose_calls: how often update_pose will be called after each successful est_base_t_cam
    """
    creator:Callable[[RobotEnvironment, TimeTracker], PosePredictor]
    name: str
    number_retries: int = 1
    max_number_consecutive_update_pose_calls: int = 0

    def __post__init__(self):
        assert isinstance(self.name, str)
        assert self.number_retries > 0
        assert self.max_number_consecutive_update_pose_calls >= 0

class NPredictors1DatasetGrader:
    def __init__(
            self,
            gradable_pose_predictors: list[GradablePosePredictor],
            robot_env:RobotEnvironment,
            headset_data:HeadsetData,
    )->None:
        """
        :param gradable_pose_predictors: A list of N gradable PosePredictors
        :param robot_env: A RobotEnvironment object which will be used to create the PosePredictors
        :param headset_data: A HeadsetData object on which the PosePredictors will be evaluated
        """
        self.gradable_pose_predictors = gradable_pose_predictors

        # Creation of n predictors
        self.predictors = []
        self.creation_subcomponent_time_trackers = []
        self.creation_time_tracker = TimeTracker()

        for gradable_pose_predictor in gradable_pose_predictors:
            self.creation_time_tracker.reset_elapsed_time()
            creation_subcomponent_time_tracker = TimeTracker()
            self.predictors.append(
                gradable_pose_predictor.creator(robot_env, creation_subcomponent_time_tracker)
            )
            self.creation_subcomponent_time_trackers.append(creation_subcomponent_time_tracker)
            self.creation_time_tracker.add_time_stamp(gradable_pose_predictor.name)


        # Run predictions
        self.headset_data = headset_data
        self.graders = []
        for predictor, gradable_pose_predictor in zip(self.predictors, gradable_pose_predictors):
            grader = PredictionOnDataset(
                predictor=predictor,
                headset_data=headset_data,
                number_consecutive_update_pose_calls=gradable_pose_predictor.max_number_consecutive_update_pose_calls,
                number_retry=gradable_pose_predictor.number_retries
            )
            self.graders.append(grader)

    def get_creation_times(self)->dict[str,tuple[float, list[tuple[str, float]]]]:
        """
        Summarizes the creation times of the PosePredictors, in a data structure.
        :return: {predictor_name: (creation_time, [(creation_subcomponent_name, creation_subcomponent_time), ...])}
        """
        times = {}
        for gpp, subc_tt in zip(self.gradable_pose_predictors, self.creation_subcomponent_time_trackers):
            times[gpp.name] = (
                self.creation_time_tracker.get_timestamp_name_avg_time(gpp.name),
                subc_tt.return_averaged_times()
            )
        return times

    def print_summary(self):
        """
        Print the summary of the PosePredictors performances
        """
        print(f"{'name':<30} {'success ratio %':<12} {'T/frame [ms]':<10} {'avg t_err [mm]':<10} {'avg r_err [deg]':<10} \n")
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            print(
                f"{gpp.name:<30} "
                f"{grader.success_ratio * 100:>10.2f} "
                f"{grader.avg_time_for_frame_prediction * 1000:>10.0f} "
                f"{grader.avg_translational_error * 1000:>12.1f} "
                f"{np.rad2deg(grader.avg_rotational_error):>12.1f}"
            )



    def plot_creation_times(self, ax):
        pass

    def plot_est_base_t_cam_times(self, ax):
        pass

    def plot_update_pose_times(self, ax):
        pass

    def _plot_times(self, ax, times:dict[str,tuple[float, list[tuple[str, float]]]]):
        pass

    def plot_frame_prediction_times(self, ax):
        pass

    def plot_hz_vs_rotational_error_deg(self, ax):
        pass

    def plot_hz_vs_translational_error_mm(self, ax):
        pass

    def _plot_hz_vs_metric(self,
                           ax,
                           hz:list[float],
                           metric:list[float],
                           names:list[str]
                           )->None:
        pass

    def visualize_predictions_3d(self):
        pass