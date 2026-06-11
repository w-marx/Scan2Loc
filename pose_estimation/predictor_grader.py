import sys, os
from typing import Callable
import open3d as o3d
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from shared.assertion_helpers import *
from shared.se3_utilities import translational_difference, rotational_difference, ate_rmse, rte_rotational_errors_rmse, rte_translational_errors_rmse
from shared.image_camera_manipulation import create_3d_camera

from robot_environment import RobotEnvironment
from headset_data import HeadsetData
from geometric_utilities.time_tracker import TimeTracker
from geometric_utilities.slam2mp4 import VideoGenerator, FeatureDrawing, InfoCard


from predictor_handling import PosePredictor


class PredictionOnDataset:
    def __init__(self,
                 predictor:PosePredictor,
                 headset_data:HeadsetData,
                 number_retry:int = 1,
                 vid_gen:VideoGenerator | None = None,
                 video_save_location:str = "test.mp4"
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

        print(f"Avg. time per prediction: {self.time_per_successful_prediction*1000}ms")
        print(f"est_base_t_cam subcomponent times:\n")
        self._est_base_t_cam_time_tracker.print_report()

        print(f"\n\nAvg. error: {np.round(self.avg_translational_error*1000,1)}mm and {np.round(np.rad2deg(self.avg_rotational_error),1)}°")
        print(f"Median. error: {np.round(self.median_translational_error*1000,1)}mm and {np.round(np.rad2deg(self.median_rotational_error),1)}°")
        print(f"ATE RMSE: {np.round(self.ate_translation_rmse * 1000,1)}mm and {np.round(np.rad2deg(self.ate_rot_rmse))}°")
        print(f"RTE RMSE: {np.round(self.rte_translation_rmse * 1000,1)}mm and {np.round(np.rad2deg(self.rte_rotational_rmse),1)}°")


    def get_subcomponent_times_est_base_t_cam_call(self)-> list[tuple[str, float]]:
        """
        Returns the avg. times and their subcomponents per est_base_t_cam call
        :return: The time per call and a list of [subcomponent_name, avg time in seconds] tuples (sorted by time descending)
        """
        complete_time = self._predictions_whole_time_tracker.get_timestamp_name_avg_time("1 est_base_t_cam call")
        sub_times = self._est_base_t_cam_time_tracker.return_averaged_times()
        return complete_time, sub_times

    def get_subcomponent_times_update_pose_call(self)->list[tuple[str, float]]:
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

    def __post__init__(self):
        assert isinstance(self.name, str)
        assert self.number_retries > 0

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
        self.robot_env = robot_env


        # Creation of n predictors
        self.creation_subcomponent_time_trackers = []
        self.creation_time_tracker = TimeTracker()
        self.headset_data = headset_data
        self.graders:list[PredictionOnDataset] = []

        for gradable_pose_predictor in gradable_pose_predictors:
            self.creation_time_tracker.reset_elapsed_time()
            creation_subcomponent_time_tracker = TimeTracker()
            predictor = gradable_pose_predictor.creator(robot_env, creation_subcomponent_time_tracker)
            self.creation_subcomponent_time_trackers.append(creation_subcomponent_time_tracker)
            self.creation_time_tracker.add_time_stamp(gradable_pose_predictor.name)

            grader = PredictionOnDataset(
                predictor=predictor,
                headset_data=headset_data,
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
        print(f"{'name':<30} {'success ratio %':<20} {'T/frame [ms]':<15} {'avg t_err [mm]':<15} {'avg r_err [deg]':<15} {'med t_err [mm]':<15} {'med r_err [deg]':<15}\n")
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            print(
                f"{gpp.name:<30} "
                f"{grader.success_ratio * 100:<20.2f} "
                f"{grader.time_per_successful_prediction * 1000:<15.0f} "
                f"{grader.avg_translational_error * 1000:<15.1f} "
                f"{np.rad2deg(grader.avg_rotational_error):<15.1f}"
                f"{grader.median_translational_error * 1000:<15.1f} "
                f"{np.rad2deg(grader.median_rotational_error):<15.1f}"
            )


    def plot_creation_times(self, ax):
        creation_times = self.get_creation_times()
        self._plot_times(ax = ax, times=creation_times, title="Creation times")

    def _plot_times(
            self, 
            ax, 
            times:dict[str,tuple[float, list[tuple[str, float]]]],
            title:str = "Time breakdown"
        ):
        labels = list(times.keys())
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels))

        all_sub_labels = set()

        total_times = []
        sub_dicts = []

        for label in labels:
            total, sub_times = times[label]
            total_times.append(total)
            sub_dicts.append(dict(sub_times))
            all_sub_labels.update(sub_dicts[-1].keys())

        for sub_label in sorted(all_sub_labels):
            values = []
            for i, label in enumerate(labels):
                values.append(sub_dicts[i].get(sub_label, 0.0)*1000)

            ax.bar(x, values, bottom=bottoms, label=sub_label)
            bottoms += np.array(values)

        # compute and plot "rest"
        total_times = np.array(total_times)*1000
        rest = total_times - bottoms
        print(f"rest: {rest}")

        ax.bar(x, rest, bottom=bottoms, label="rest", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Time [ms]")
        ax.set_title(title)
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)


    def plot_successful_frame_prediction_times(self, ax):
        names = []
        times = []
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            time = grader.time_per_successful_prediction
            if time is not None:
                names.append(gpp.name) 
                times.append(time*1000)

        sorted_pairs = sorted(zip(names, times), key=lambda x: x[1])
        names, times = zip(*sorted_pairs)

        ax.bar(names, times)

        ax.set_xlabel("Predictor")
        ax.set_ylabel("Avg time per successful prediction [ms]")
        ax.tick_params(axis='x', rotation=45)

        

    def plot_hz_vs_rotational_error_deg(self, ax):
        pass

    def plot_hz_vs_translational_error_mm(self, ax):
        pass

    def plot_translational_errors(self, ax):
        for gpp, grader in zip(self.gradable_pose_predictors,self.graders):
            timed_translat_errors = grader.timed_translational_errors
            name = gpp.name
            times = [i for i, _ in timed_translat_errors]
            errors_mm = [e*1000 for _, e in timed_translat_errors]
            ax.plot(times, errors_mm, label = f"{name} avg: {np.round(grader.avg_translational_error*1000,1)}", alpha = 1.0)

        ax.set_title("Translational errors over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Translational log error [mm]")
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()

    def plot_rotational_errors(self, ax):
        for gpp, grader in zip(self.gradable_pose_predictors,self.graders):
            times = [i for i, _ in grader.timed_rotational_errors]
            errors_deg = [np.rad2deg(e) for _, e in grader.timed_rotational_errors]
            ax.plot(times, errors_deg, label = f"{gpp.name} avg: {np.round(np.rad2deg(grader.avg_rotational_error), 1)}", alpha = 1.0)

        ax.set_title("Rotational errors over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Rotational log error [deg]")
        ax.set_yscale('log')
        ax.grid(True)
        ax.legend()
        



    def _plot_hz_vs_metric(self,
                           ax,
                           hz:list[float],
                           metric:list[float],
                           names:list[str]
                           )->None:
        pass

    def visualize_predictions_3d(self):
        """
        Visualizes the predictions made by the predictors using open3d
        :param robot_env: RobotEnvironment or None, if not None will be added to the plot
        :param show_label: whether to show the label camera frames or not
        """
        to_vis = self.robot_env.visualize_3d_data(visualize=False)

        colors = plt.cm.plasma(np.linspace(0, 1, len(self.graders)))[:, :3]
        colors = [[0, 1.0, 0]] + list(colors)

        trajectories = [[b_t_h for b_t_h in self.headset_data.robot_base_t_headset_s if b_t_h is not None]]
        trajectories += [g.predicted_base_t_headset_s_no_none for g in self.graders]


        for i, trajectory in enumerate(trajectories):
            if len(trajectory) == 0:
                continue

            traj_line_set = o3d.geometry.LineSet()
            traj_line_set.points = o3d.utility.Vector3dVector(np.asarray(trajectory)[:,:3,3])
            traj_line_set.lines = o3d.utility.Vector2iVector([[j, j+1] for j in range(len(trajectory)-1)])
            traj_line_set.paint_uniform_color(colors[i])
            to_vis.append(traj_line_set)

        o3d.visualization.draw_geometries(to_vis, f"Predicted trajectories visualisation")