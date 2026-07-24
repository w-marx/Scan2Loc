from typing import Callable, SupportsFloat, Literal
import open3d as o3d
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from enum import Enum
from tqdm import tqdm


from ..data_interfaces.scanned_3d_environment import Scanned3dEnvironment
from ..data_interfaces.headset_recording import HeadsetRecording

from ..utilities.time_tracker import TimeTracker

from .prediction_on_dataset import PredictionOnDataset, format_optional, fmt_mae, fmt_median, fmt_rmse
from .headset_localizer import HeadsetLocalizer
from .ray_intersection_error import FastRayIntersectionError

@dataclass(frozen=True, kw_only=True)
class GradableLocalizer:
    """
    A dataclass to grade a Pose Predictor including its runtime
    :param creator: A function that takes an Robot environment and a TimeTracker for the init and returns the HeadsetLocalizer
    :param name: The identifying name of the HeadsetLocalizer (may appear on plots)
    :param number_retries: The number of times to retries passed to est_base_t_cam
    :param category: A category for the predictor (for most plots Category-Name can be used if wanted)
    :param index: An optional value for plotting different predictors on an axis
    """
    creator:Callable[[Scanned3dEnvironment, TimeTracker], HeadsetLocalizer]
    name: str
    number_retries: int = 1
    category: str = ""
    value:None | SupportsFloat = None

    def __post_init__(self):
        assert isinstance(self.name, str)
        assert self.number_retries > 0

    @property
    def c_name(self):
        return f"{self.category}-{self.name}" if self.category != "" else self.name



class SingleValueErrorType(Enum):
    AVG_TRANSLATIONAL = "Avg. Translational Error [mm]"
    MED_TRANSLATIONAL = "Med. Translational Error [mm]"
    AVG_ROTATIONAL = "Avg. Rotational Error [deg]"
    MED_ROTATIONAL = "Med. Rotational Error [deg]"
    ATE_RMSE_TRANSLATIONAL = "ATE RMSE [mm]"
    ATE_RMSE_ROTATIONAL = "ATE RMSE [deg]"
    RTE_RMSE_TRANSLATIONAL = "RTE RMSE [mm]"
    RTE_RMSE_ROTATIONAL = "RTE RMSE [deg]"
    SUCCESS_RATE = "Success Rate [%]"
    AVG_NUMBER_POINTS_INLIERS = "Average number of PnP inliers"
    AVG_NUMBER_OF_TRIES = "Average number of Point extract & match tries"
    AVG_GRIPPING_ERROR = "Average translational error for point grip"
    MEDIAN_GRIPPING_ERROR = "Median translational error for point grip"

    
    @property
    def calculator(self) -> Callable[["PredictionOnDataset"], SupportsFloat | None]:
        return SINGLE_VALUE_ERROR_CALCULATORS[self]
    
    @property
    def ylabel(self) -> str:
        return SINGLE_VALUE_ERROR_LABELS[self]


SINGLE_VALUE_ERROR_CALCULATORS:dict[SingleValueErrorType, Callable[[PredictionOnDataset], SupportsFloat | None]] = {
    SingleValueErrorType.AVG_TRANSLATIONAL: lambda grader: (
        None if grader.avg_translational_error is None 
        else grader.avg_translational_error * 1000
    ),
    SingleValueErrorType.MED_TRANSLATIONAL: lambda grader: (
        None if grader.median_translational_error is None 
        else grader.median_translational_error * 1000
    ),
    SingleValueErrorType.AVG_ROTATIONAL: lambda grader: (
        None if grader.avg_rotational_error is None 
        else np.rad2deg(grader.avg_rotational_error)
    ),
    SingleValueErrorType.MED_ROTATIONAL: lambda grader: (
        None if grader.median_rotational_error is None 
        else np.rad2deg(grader.median_rotational_error)
    ),
    SingleValueErrorType.ATE_RMSE_TRANSLATIONAL: lambda grader: (
        None if grader.ate_translation_rmse is None 
        else grader.ate_translation_rmse * 1000
    ),
    SingleValueErrorType.ATE_RMSE_ROTATIONAL: lambda grader: (
        None if grader.ate_rot_rmse is None 
        else np.rad2deg(grader.ate_rot_rmse)
    ),
    SingleValueErrorType.RTE_RMSE_TRANSLATIONAL: lambda grader: (
        None if grader.rte_translation_rmse is None 
        else grader.rte_translation_rmse * 1000
    ),
    SingleValueErrorType.RTE_RMSE_ROTATIONAL: lambda grader: (
        None if grader.rte_rotational_rmse is None 
        else np.rad2deg(grader.rte_rotational_rmse)
    ),
    SingleValueErrorType.SUCCESS_RATE: lambda grader: (
        None if grader.success_ratio is None 
        else grader.success_ratio * 100
    ),
    SingleValueErrorType.AVG_NUMBER_POINTS_INLIERS: lambda grader: (
        None if grader.avg_number_of_inliers is None 
        else grader.avg_number_of_inliers
    ),
    SingleValueErrorType.AVG_NUMBER_OF_TRIES: lambda grader: (
        None if grader.avg_number_of_tries is None 
        else grader.avg_number_of_tries
    ),
    SingleValueErrorType.AVG_GRIPPING_ERROR: lambda grader: (
        None if grader.avg_gripping_error is None 
        else grader.avg_gripping_error * 1000
    ),
    SingleValueErrorType.MEDIAN_GRIPPING_ERROR: lambda grader: (
        None if grader.median_gripping_error is None 
        else grader.median_gripping_error * 1000
    ),
    
}

SINGLE_VALUE_ERROR_LABELS = {
    SingleValueErrorType.AVG_TRANSLATIONAL: "Error [mm]",
    SingleValueErrorType.MED_TRANSLATIONAL: "Error [mm]",
    SingleValueErrorType.AVG_ROTATIONAL: "Error [deg]",
    SingleValueErrorType.MED_ROTATIONAL: "Error [deg]",
    SingleValueErrorType.ATE_RMSE_TRANSLATIONAL: "Error [mm]",
    SingleValueErrorType.ATE_RMSE_ROTATIONAL: "Error [deg]",
    SingleValueErrorType.RTE_RMSE_TRANSLATIONAL: "Error [mm]",
    SingleValueErrorType.RTE_RMSE_ROTATIONAL: "Error [deg]",
    SingleValueErrorType.SUCCESS_RATE: "Rate [%]",
    SingleValueErrorType.AVG_NUMBER_POINTS_INLIERS: "Number point inliers [1]",
    SingleValueErrorType.AVG_NUMBER_OF_TRIES: "Number of tries [1]",
    SingleValueErrorType.AVG_GRIPPING_ERROR: "Error [mm]",
    SingleValueErrorType.MEDIAN_GRIPPING_ERROR: "Error [mm]"
}


class TimeSeriesErrorType(Enum):
    ABS_TRANSLATIONAL = "Absolute Translational error [mm]"
    ABS_ROTATIONAL = "Absolute Rotational error [mm]"
    ABS_GRIPPING = "Absolute point gripping error [mm]"
    RTE_TRANSLATIONAL = "Relative translational error [mm]"
    RTE_ROTATIONAL = "Relative rotational error [deg]"
    
    @property
    def calculator(self) -> Callable[[PredictionOnDataset], list[tuple[int, float]]]:
        return TIME_SERIES_ERROR_CALCULATORS[self]
    
    @property
    def ylabel(self) -> str:
        return TIME_SERIES_ERROR_LABELS[self]


TIME_SERIES_ERROR_CALCULATORS:dict[TimeSeriesErrorType, Callable[[PredictionOnDataset], list[tuple[int, float]]]] = {
    TimeSeriesErrorType.ABS_TRANSLATIONAL: lambda grader: (
        [(idx, error*1000) for idx,error in grader.timed_translational_errors]
    ),
    TimeSeriesErrorType.ABS_ROTATIONAL: lambda grader: (
        [(idx, np.rad2deg(error)) for idx,error in grader.timed_rotational_errors]
    ),
    TimeSeriesErrorType.ABS_GRIPPING: lambda grader: (
        [(idx, error*1000) for idx,error in grader.timed_gripping_errors]
    ),
    TimeSeriesErrorType.RTE_TRANSLATIONAL: lambda grader: (
        [(idx, error*1000) for idx,error in enumerate(grader.rte_translation_errors) if np.isfinite(error)]
    ),
    TimeSeriesErrorType.RTE_ROTATIONAL: lambda grader: (
        [(idx, np.rad2deg(error)) for idx,error in enumerate(grader.rte_rotational_errors) if np.isfinite(error)]
    ),
}

TIME_SERIES_ERROR_LABELS = {
    TimeSeriesErrorType.ABS_TRANSLATIONAL: "Error [mm]",
    TimeSeriesErrorType.ABS_ROTATIONAL: "Error [deg]",
    TimeSeriesErrorType.ABS_GRIPPING: "Error [mm]",
    TimeSeriesErrorType.RTE_TRANSLATIONAL: "Error [mm]",
    TimeSeriesErrorType.RTE_ROTATIONAL: "Error [deg]"
}


class NPredictors1DatasetGrader:
    def __init__(
            self,
            gradable_pose_predictors: list[GradableLocalizer],
            robot_env:Scanned3dEnvironment,
            headset_data:HeadsetRecording,
            use_tqdm_for_predictors:bool = False,
            use_tqdm_for_frames:bool = True,
            compute_gripping_error:bool = False,
    )->None:
        """
        :param gradable_pose_predictors: A list of N gradable PosePredictors
        :param robot_env: A RobotEnvironment object which will be used to create the PosePredictors
        :param headset_data: A HeadsetData object on which the PosePredictors will be evaluated
        """
        self.gradable_pose_predictors = gradable_pose_predictors
        self.robot_env = robot_env

        if compute_gripping_error:
            gripping_error_calculator = FastRayIntersectionError(points=robot_env.robot_xyz_images, intrinsics=headset_data.intrinsic_cam_mtx)
        else:
            gripping_error_calculator = None


        # Creation of n predictors
        self.creation_subcomponent_time_trackers = []
        self.creation_time_tracker = TimeTracker()
        self.headset_data = headset_data
        self.graders:list[PredictionOnDataset] = []

        for gradable_pose_predictor in (tqdm(gradable_pose_predictors) if use_tqdm_for_predictors else gradable_pose_predictors):
            self.creation_time_tracker.reset_elapsed_time()
            creation_subcomponent_time_tracker = TimeTracker()
            predictor = gradable_pose_predictor.creator(robot_env, creation_subcomponent_time_tracker)
            self.creation_subcomponent_time_trackers.append(creation_subcomponent_time_tracker)
            self.creation_time_tracker.add_time_stamp(gradable_pose_predictor.c_name)

            grader = PredictionOnDataset(
                predictor=predictor,
                headset_data=headset_data,
                number_retry=gradable_pose_predictor.number_retries,
                gripping_error=gripping_error_calculator,
                use_tqdm=use_tqdm_for_frames
            )
            self.graders.append(grader)


    def get_creation_times(self)->pd.DataFrame:
        """
        Summarizes the creation times of the PosePredictors, in a data structure.
        :return: {predictor_name: (creation_time, [(creation_subcomponent_name, creation_subcomponent_time), ...])}
        """
        rows = {}

        for gpp, subc_tt in zip(
            self.gradable_pose_predictors,
            self.creation_subcomponent_time_trackers,
        ):
            total = self.creation_time_tracker.get_timestamp_name_avg_time(gpp.c_name)
            if total is None:
                continue

            sub_times = subc_tt.return_averaged_times()

            row = {name: value * 1000 for name, value in sub_times}

            subtotal = sum(value for _, value in sub_times)
            row["rest"] = (total - subtotal) * 1000

            rows[gpp.c_name] = row

        return pd.DataFrame.from_dict(rows, orient="index").fillna(0)
    
    
    def get_prediction_times(self)->pd.DataFrame:
        """
        Summarizes the prediction times of the PosePredictors, in a data structure.
        :return: {predictor_name: (pred_time, [(creation_subcomponent_name, prediction_subcomponent_time), ...])}
        """
        rows = {}
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            total, sub_times = grader.get_prediction_times()

            if total is None:
                continue

            row = {name: value * 1000 for name, value in sub_times}

            subtotal = sum(value for _, value in sub_times)
            row["rest"] = (total - subtotal) * 1000

            rows[gpp.c_name] = row

        return pd.DataFrame.from_dict(rows, orient="index").fillna(0)
    

    def print_error_under_limits(self, limits_m:list[float], error_type:Literal['ATE', 'ARE', 'RIE']):

        rows = []
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            dict = {"Name": gpp.c_name}
            err_s = {
                'ATE':grader.translational_errors,
                'ARE':grader.rotational_errors,
                'RIE':grader.gripping_error_s
            }
            t_error_s = err_s[error_type]
            if t_error_s is None:
                continue
            t_error_s = np.array(t_error_s)

            for limit in limits_m:
                limit_mask = t_error_s < limit
                valid_errors = t_error_s[limit_mask]
                avg_error = fmt_mae(valid_errors, fmt=".4f")
                success = format_optional(np.sum(limit_mask)/grader.number_attempted_predictions, factor=100, fmt="5.1f")
                dict[str(limit)] = avg_error+f"m, {success}%"
            
            rows.append(dict)
                
        df = pd.DataFrame(rows)
        print(f"{error_type} under limits")
        print(df.to_string(index=False))


    def print_summary(self):
        """
        Print a summary of the PosePredictor performances.
        """
        rows = []
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            grader.translational_errors
            rows.append({
                "Name": gpp.c_name,
                "Success [%]": format_optional(grader.success_ratio, factor=100),
                "T/frame [ms]": format_optional(grader.time_per_successful_prediction, factor=1000),

                "Avg ATE [mm]": fmt_mae(errors=grader.translational_errors, factor=1000),
                "Avg ARE [deg]": fmt_mae(grader.rotational_errors, factor=180/np.pi, fmt=".1f"),

                "Med ATE [mm]": fmt_median(grader.translational_errors, factor=1000, fmt=".1f"),
                "Med ARE [deg]": fmt_median(grader.rotational_errors, factor=180/np.pi, fmt=".1f"),

                "RMSE ATE [mm]":fmt_rmse(grader.translational_errors, factor=1000, fmt=".1f"),
                "RMSE ARE [deg]":fmt_rmse(grader.rotational_errors, factor=180/np.pi, fmt=".1f"),

                "Avg RIE [mm]": fmt_mae(grader.gripping_error_s, factor=1000, fmt=".1f"),
                "Med RIE [mm]": fmt_median(grader.gripping_error_s, factor=1000, fmt=".1f"),
                "RMSE RIE [mm]": fmt_rmse(grader.gripping_error_s, factor=1000, fmt=".1f")
            })

        df = pd.DataFrame(rows)

        print(df.to_string(index=False))


    @staticmethod
    def plot_times(
        ax,
        time_df: pd.DataFrame,
        title: str,
        plot_legend: bool = True,
        rotate_x_labels:int|None = 45
    ):
        cols = sorted(c for c in time_df.columns if c != "rest")
        if "rest" in time_df.columns:
            cols.append("rest")

        time_df = time_df[cols]

        time_df.plot(kind="bar", stacked=True, ax=ax, rot=0, legend=plot_legend)

        ax.set_ylabel("Time [ms]")
        ax.set_title(title)

        if plot_legend:
            ax.legend()
        if rotate_x_labels is not None:
            plt.setp(ax.get_xticklabels(), rotation=rotate_x_labels, ha="right")

        ax.grid(True, axis="y", alpha=0.3)



    def plot_creation_times(self, ax:Axes, plot_legend:bool = True):
        NPredictors1DatasetGrader.plot_times(
            ax=ax,
            time_df=self.get_creation_times(),
            title="Predictor Creation Times",
            plot_legend=plot_legend
        )


    def print_creation_times(self):
        print(self.get_creation_times().to_string(float_format="{:.1f}".format))


    def plot_prediction_times(self, ax:Axes, plot_legend:bool = True, rotate_x_labels:int | None = 45):
        NPredictors1DatasetGrader.plot_times(
            ax=ax,
            time_df=self.get_prediction_times(),
            title="Average time consumpion for a sucessful prediction",
            plot_legend=plot_legend,
            rotate_x_labels=rotate_x_labels
        )


    def print_prediction_times(self):
        print(self.get_prediction_times().to_string(float_format="{:.1f}".format))


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
    

    @staticmethod
    def _plot_frontier_plot(
        ax:Axes,
        df: pd.DataFrame,
        xkey:str = "Hz [1/s]",
        ykey:str = "Translational error [mm]",
        title:str = "FPS vs translational MAE",
        plot_frontier:bool = False, 
        use_category:bool = False,
        invert_y:bool = True,
        invert_x:bool = False,
        category_key:str = "Category",
        name_key:str = "Name",
        full_name_key:str = "Predictor",
        plot_legend:bool = True,
        plot_names:bool = True,
        in_plot_text_size:int = 10,
        adjust_texts:bool = True
    ):
        hue_key = category_key if use_category else full_name_key
        palette = sns.color_palette("tab10", n_colors=df[hue_key].nunique())
        color_map = dict(zip(sorted(df[hue_key].unique()), palette))

        sns.scatterplot(data=df, x=xkey, y=ykey, hue=hue_key,
            palette=color_map, ax=ax, s=80, alpha=0.7, legend=plot_legend
        )

        if plot_frontier:
            pareto_df = NPredictors1DatasetGrader.compute_pareto_frontier(df, key1=xkey, key2=ykey,  smaller_better_2=invert_y)
            pareto_df = pareto_df.sort_values(xkey)

            ax.plot(pareto_df[xkey], pareto_df[ykey], color="black", linewidth=1, linestyle = "--", alpha=0.4, label="Frontier")

        if use_category:
            for category, cat_df in df.groupby("Category"):
                cat_df = cat_df.sort_values(xkey)
                ax.plot(cat_df[xkey], cat_df[ykey], linewidth=1, alpha=0.5, color = color_map[category])

        if plot_names:
            y_offset = 0 if adjust_texts else (df[ykey].max() - df[ykey].min()) * 0.03
            texts = []
            for _, row in df.iterrows():
                texts.append(
                    ax.text(
                        row[xkey],
                        row[ykey]+y_offset,
                        (row[name_key] if use_category else row[full_name_key]),
                        fontsize=in_plot_text_size
                    )
                )
            if adjust_texts:
                adjust_text(texts, ax=ax, force_text=(0.5, 0.5), arrowprops=dict(arrowstyle="-", lw=1, alpha=0.0))

        ax.margins(x=0.15, y=0.2)

        if invert_y:
            ax.invert_yaxis()

        if invert_x:
            ax.invert_xaxis()

        ax.set_title(title)
        if plot_legend:
            ax.legend()


    def plot_hz_vs_error(
            self, 
            ax:Axes, 
            error_type:SingleValueErrorType,
            custom_title:str | None = None,
            plot_frontier:bool = False, 
            use_category:bool = False,
            invert_y:bool = True,
            plot_legend:bool = True,
            plot_names:bool = True,
            adjust_texts:bool = True
        ):
        data = []

        error_access = error_type.calculator
        error_ylabel = error_type.ylabel

        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            time = grader.time_per_successful_prediction
            if time is not None:
                metric = error_access(grader)
                if metric is not None:
                    data.append({
                        "Predictor": gpp.c_name,
                        "Category": gpp.category,
                        "Name":gpp.name,
                        error_ylabel: metric,
                        "Hz [1/s]": 1 / time
                    })

        df = pd.DataFrame(data)

        if df.empty:
            return

        self._plot_frontier_plot(
            ax = ax,
            df = df,
            xkey = "Hz [1/s]",
            ykey = error_ylabel,
            title = f"FPS vs {error_type.value}" if custom_title is None else custom_title,
            plot_frontier = plot_frontier, 
            use_category = use_category,
            invert_y = invert_y,
            category_key = "Category",
            name_key = "Name",
            full_name_key = "Predictor",
            plot_legend = plot_legend,
            plot_names = plot_names,
            adjust_texts = adjust_texts
        )
    
    
    def plot_error_vs_error(
            self, 
            ax:Axes, 
            error_type_1:SingleValueErrorType,
            error_type_2:SingleValueErrorType,
            custom_title:str | None = None,
            plot_legend:bool = True,
            plot_frontier:bool = False, 
            use_category:bool = False,
            invert_x:bool = False,
            invert_y:bool = False,
            use_in_plot_text:bool = True,
            adjust_texts:bool = True
        ):
        data = []

        error1_access = error_type_1.calculator
        error1_label = error_type_1.ylabel

        error2_access = error_type_2.calculator
        error2_label = error_type_2.ylabel

        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            time = grader.time_per_successful_prediction
            if time is not None:
                metric1 = error1_access(grader)
                metric2 = error2_access(grader)
                if metric1 is not None and metric2 is not None:
                    data.append({
                        "Predictor": gpp.c_name,
                        "Category": gpp.category,
                        "Name":gpp.name,
                        error1_label: metric1,
                        error2_label: metric2
                    })

        df = pd.DataFrame(data)

        if df.empty:
            return

        self._plot_frontier_plot(
            ax = ax,
            df = df,
            xkey = error1_label,
            ykey = error2_label,
            title = f"{error_type_1.value} vs {error_type_2.value}" if custom_title is None else custom_title,
            plot_frontier = plot_frontier, 
            use_category = use_category,
            invert_y = invert_y,
            category_key = "Category",
            name_key = "Name",
            full_name_key = "Predictor",
            invert_x = invert_x,
            plot_names = use_in_plot_text,
            plot_legend = plot_legend,
            adjust_texts = adjust_texts
        )


    def plot_value_vs_error(
            self,
            ax:Axes,
            x_axis_label:str,
            x_axis_title_name:str,
            error_type:SingleValueErrorType,
            plot_frontier:bool = False,
            use_category:bool = False,
            plot_legend:bool = False,
            plot_names:bool = False,
            adjust_texts:bool = True
        ):
        data = []

        error_access = error_type.calculator
        error_label = error_type.ylabel

        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            time = grader.time_per_successful_prediction
            if time is not None:
                x_index = gpp.value
                metric = error_access(grader)

                if metric is not None and x_index is not None:
                    data.append({
                        "Predictor": gpp.c_name,
                        "Category": gpp.category,
                        "Name":gpp.name,
                        x_axis_label: x_index,
                        error_label: metric
                    })

        df = pd.DataFrame(data)

        self._plot_frontier_plot(
            ax = ax,
            df = df,
            xkey = x_axis_label,
            ykey = error_label,
            title = f"{x_axis_title_name} vs {error_type.value}",
            plot_frontier = plot_frontier,
            use_category = use_category,
            invert_y = False,
            category_key = "Category",
            name_key = "Name",
            full_name_key = "Predictor",
            plot_legend=plot_legend,
            plot_names = plot_names,
            adjust_texts = adjust_texts
        )


    def plot_time_series_error(self, ax:Axes, error_type:TimeSeriesErrorType, use_log_scale:bool = False, fmt = ".1f"):
        data = []
        for gpp, grader in zip(self.gradable_pose_predictors, self.graders):
            error_dict = {frame: error for frame, error in error_type.calculator(grader)}
            
            only_error_s = [error for _, error in error_type.calculator(grader)]
            avg_error = format_optional(np.mean(only_error_s), fmt=fmt)

            predictor_name = f"{gpp.c_name} (avg: {avg_error})"


            for frame in range(self.headset_data.n_frames):
                error = error_dict.get(frame, np.nan)
                data.append({'Frame': frame, error_type.ylabel: error, 'Predictor': predictor_name})
        
        df = pd.DataFrame(data)
        df = df.sort_values(['Predictor','Frame'])
        
        for predictor, group in df.groupby('Predictor'):
            group = group.sort_values('Frame')

            ax.plot(group['Frame'], group[error_type.ylabel], label=predictor, linewidth=2)

        ax.legend()

        if use_log_scale:
            ax.set_yscale("log")

        ax.set_title(f"{error_type.value} over time")
        ax.set_xlabel("Frame")
        ax.set_ylabel(error_type.ylabel)
        

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