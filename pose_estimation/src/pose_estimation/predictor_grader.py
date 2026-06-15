from typing import Callable
import open3d as o3d
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from adjustText import adjust_text


from .robot_environment import RobotEnvironment
from .headset_data import HeadsetData
from .geometric_utilities.time_tracker import TimeTracker
from .geometric_utilities.slam2mp4 import VideoGenerator, FeatureDrawing, InfoCard
from .prediction_on_dataset import *
from .predictor_handling import PosePredictor


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
    category: str = ""

    def __post_init__(self):
        assert isinstance(self.name, str)
        assert self.number_retries > 0

    @property
    def c_name(self):
        return f"{self.category}-{self.name}"


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
            self.creation_time_tracker.add_time_stamp(gradable_pose_predictor.c_name)

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
            times[gpp.c_name] = (
                self.creation_time_tracker.get_timestamp_name_avg_time(gpp.c_name),
                subc_tt.return_averaged_times()
            )
        return times

    def print_summary(self):
        """
        Print a summary of the PosePredictor performances.
        """
        rows = []

        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            rows.append({
                "Name": gpp.c_name,
                "Success [%]": grader.success_ratio * 100,
                "T/frame [ms]": (
                    grader.time_per_successful_prediction * 1000
                    if grader.time_per_successful_prediction is not None
                    else np.nan
                ),
                "Avg t_err [mm]": grader.avg_translational_error * 1000,
                "Avg r_err [deg]": np.rad2deg(grader.avg_rotational_error),
                "Med t_err [mm]": grader.median_translational_error * 1000,
                "Med r_err [deg]": np.rad2deg(grader.median_rotational_error),
            })

        df = pd.DataFrame(rows)

        print(
            df.to_string(
                index=False,
                formatters={
                    "Success [%]": "{:.2f}".format,
                    "T/frame [ms]": "{:.0f}".format,
                    "Avg t_err [mm]": "{:.1f}".format,
                    "Avg r_err [deg]": "{:.1f}".format,
                    "Med t_err [mm]": "{:.1f}".format,
                    "Med r_err [deg]": "{:.1f}".format,
                },
            )
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
                    'Predictor': gpp.c_name,
                    'Time [ms]':time * 1000
                })
        
        df = pd.DataFrame(data)
        order = df.sort_values('Time [ms]', ascending=False)['Predictor']
        sns.barplot(data=df, y='Predictor', x='Time [ms]', order=order, ax=ax)
        ax.set_ylabel("Predictor")
        ax.set_xlabel("Avg time per successful prediction [ms]")
        ax.margins(y=0.15)


    @staticmethod
    def compute_pareto_frontier(
            df, 
            key1:str = "Hz [1/s]", 
            smaller_better_1:bool = False,
            key2:str = "Error [deg]", 
            smaller_better_2:bool = True, 
        ):
        df_sorted = df.sort_values(by=key1, ascending=smaller_better_1)

        pareto = []
        best_error = float("inf") if smaller_better_2 else -float("inf")
        for _, row in df_sorted.iterrows():
            if (smaller_better_2 and row[key2] < best_error) or (not smaller_better_2 and row[key2] > best_error):
                pareto.append(row)
                best_error = row[key2]

        return pd.DataFrame(pareto)


    def plot_hz_vs_rotational_error_deg(self, ax, plot_frontier = False):
        data = []

        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            time = grader.time_per_successful_prediction
            if time is not None:
                data.append({
                    "Predictor": gpp.c_name,
                    "Error [deg]": np.rad2deg(grader.avg_rotational_error),
                    "Hz [1/s]": 1 / time
                })

        df = pd.DataFrame(data)

        sns.scatterplot(data=df,
            x="Hz [1/s]",
            y="Error [deg]",
            hue="Predictor",
            ax=ax,
            s=80,
            alpha=0.7
        )

        if plot_frontier:
            pareto_df = self.compute_pareto_frontier(df)
            pareto_df = pareto_df.sort_values("Hz [1/s]")

            ax.plot(
                pareto_df["Hz [1/s]"],
                pareto_df["Error [deg]"],
                color="black",
                linewidth=1,
                alpha=0.4,
                label="Frontier"
            )

        texts = []

        for _, row in df.iterrows():
            texts.append(
                ax.text(
                    row["Hz [1/s]"],
                    row["Error [deg]"],
                    row["Predictor"],
                    fontsize=8
                )
            )

        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.5)
        )
        ax.margins(x=0.15, y=0.2)
        ax.invert_yaxis()
        ax.set_title("FPS vs Rotational Error")
        ax.legend()


    def plot_hz_vs_translational_error_mm(self, ax, plot_frontier:bool = False, use_category:bool = False):
        data = []

        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            time = grader.time_per_successful_prediction
            if time is not None:
                data.append({
                    "Predictor": gpp.c_name,
                    "Name": gpp.name,
                    "Translational error [mm]": grader.avg_translational_error*1000,
                    "Hz [1/s]": 1 / time,
                    "Category":gpp.category
                })

        df = pd.DataFrame(data)

        hue_key = "Category" if use_category else "Predictor"
        import seaborn as sns
        palette = sns.color_palette("tab10", n_colors=df[hue_key].nunique())
        color_map = dict(zip(sorted(df[hue_key].unique()), palette))


        sns.scatterplot(data=df,
            x="Hz [1/s]",
            y="Translational error [mm]",
            hue=hue_key,
            palette=color_map,
            ax=ax,
            s=80,
            alpha=0.7
        )

        if plot_frontier:
            pareto_df = self.compute_pareto_frontier(df, key2="Translational error [mm]")
            pareto_df = pareto_df.sort_values("Hz [1/s]")

            ax.plot(
                pareto_df["Hz [1/s]"],
                pareto_df["Translational error [mm]"],
                color="black",
                linewidth=1,
                alpha=0.4,
                label="Frontier"
            )

        if use_category:
            for category, cat_df in df.groupby("Category"):
                cat_df = cat_df.sort_values("Hz [1/s]")

                ax.plot(
                    cat_df["Hz [1/s]"],
                    cat_df["Translational error [mm]"],
                    linewidth=1,
                    alpha=0.5,
                    color = color_map[category],
                )

        texts = []
        for _, row in df.iterrows():
            texts.append(
                ax.text(
                    row["Hz [1/s]"],
                    row["Translational error [mm]"],
                    (row["Name"] if use_category else row["Predictor"]),
                    fontsize=8
                )
            )

        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", lw=0.5, alpha=0.5)
        )


        ax.margins(x=0.15, y=0.2)
        ax.invert_yaxis()
        ax.set_title("FPS vs translational MAE")
        ax.legend()


    def plot_translational_errors(self, ax, use_log_scale:bool = False):
        data = []
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            avg_error = np.round(grader.avg_translational_error * 1000, 1)
            predictor_name = f"{gpp.c_name} (avg: {avg_error})"

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

        if use_log_scale:
            ax.set_yscale("log")

        ax.legend()
        ax.set_title("Translational errors over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Translational error [mm]")


    def plot_rotational_errors(self, ax, use_log_scale:bool = False):
        data = []
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            avg_error = np.round(np.rad2deg(grader.avg_rotational_error), 1)
            predictor_name = f"{gpp.c_name} (avg: {avg_error})"

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

        if use_log_scale:
            ax.set_yscale("log")

        ax.set_title("Rotational errors over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel("Rotational error [deg]")
        

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