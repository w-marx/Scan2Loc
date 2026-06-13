from typing import Callable
import open3d as o3d
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from robot_environment import RobotEnvironment
from headset_data import HeadsetData
from geometric_utilities.time_tracker import TimeTracker
from geometric_utilities.slam2mp4 import VideoGenerator, FeatureDrawing, InfoCard
from prediction_on_dataset import *
from predictor_handling import PosePredictor


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
    category: None | str = None

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
        
        labels = list(creation_times.keys())
        x = np.arange(len(labels))
        bottoms = np.zeros(len(labels))

        all_sub_labels = set()

        total_times = []
        sub_dicts = []

        for label in labels:
            total, sub_times = creation_times[label]
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
        ax.set_title("Predictor Creation Times")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)


    def plot_successful_frame_prediction_times(self, ax):
        data = []
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            time = grader.time_per_successful_prediction
            if time is not None:
                data.append({
                    'Predictor': gpp.name,
                    'Time [ms]':time * 1000
                })
        
        df = pd.DataFrame(data)
        order = df.sort_values('Time [ms]', ascending=False)['Predictor']
        sns.barplot(data=df, y='Predictor', x='Time [ms]', order=order, ax=ax)
        ax.set_ylabel("Predictor")
        ax.set_xlabel("Avg time per successful prediction [ms]")
        ax.margins(y=0.15)

        
    def plot_hz_vs_rotational_error_deg(self, ax):
        pass

    def plot_hz_vs_translational_error_mm(self, ax):
        pass

    def plot_translational_errors(self, ax):
        data = []
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            avg_error = np.round(grader.avg_translational_error * 1000, 1)
            predictor_name = f"{gpp.name} (avg: {avg_error})"

            error_dict = {frame: error for frame, error in grader.timed_translational_errors}

            for frame in range(self.headset_data.n_frames):
                error = error_dict.get(frame, np.nan)
                data.append({
                    'Frame': frame,
                    'Error [mm]': error * 1000 if not np.isnan(error) else np.nan,
                    'Predictor': predictor_name
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values(['Predictor','Frame'])
        
        for predictor, group in df.groupby('Predictor'):
            group = group.sort_values('Frame')

            ax.plot(
                group['Frame'],
                group['Error [mm]'],
                label=predictor,
                linewidth=2
            )

        ax.legend()

        ax.set_title("Translational errors over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Translational error [mm]")


    def plot_rotational_errors(self, ax):
        data = []
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            avg_error = np.round(np.rad2deg(grader.avg_rotational_error), 1)
            predictor_name = f"{gpp.name} (avg: {avg_error})"

            error_dict = {frame: error for frame, error in grader.timed_rotational_errors}

            for frame in range(self.headset_data.n_frames):
                error = error_dict.get(frame, np.nan)
                data.append({
                    'Frame': frame,
                    'Error [deg]': np.rad2deg(error) if not np.isnan(error) else np.nan,
                    'Predictor': predictor_name
                })
        
        df = pd.DataFrame(data)
        df = df.sort_values(['Predictor','Frame'])
        
        for predictor, group in df.groupby('Predictor'):
            group = group.sort_values('Frame')

            ax.plot(
                group['Frame'],
                group['Error [deg]'],
                label=predictor,
                linewidth=2
            )

        ax.legend()

        ax.set_title("Rotational errors over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Rotational error [deg]")
        



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