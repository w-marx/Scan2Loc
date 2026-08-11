import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from numbers import Real
from typing import Sequence
from matplotlib.axes import Axes
import pandas as pd
import seaborn as sns

from shared.se3_utilities import translational_difference, rotational_difference, ate_rmse, rte_rotational_errors_rmse, rte_translational_errors_rmse
from shared.image_camera_manipulation import create_3d_camera

from ..data_interfaces.scanned_3d_environment import Scanned3dEnvironment
from ..data_interfaces.headset_recording import HeadsetRecording
from ..utilities.time_tracker import TimeTracker

from ..utilities.slam2mp4 import VideoGenerator, InfoCard
from .ray_intersection_error import sample_pixel_neighborhood, FastRayIntersectionError
from .headset_localizer import HeadsetLocalizer


def format_optional(value:float | int | np.number | None, fmt=".1f", default = "N/A", factor:float = 1.0)->str:
    if value is None:
        return default
    return f"{(value*factor):{fmt}}"


def clean_errors(errors:Sequence[float| None] | None | np.ndarray)->np.ndarray | None:
    if errors is None or len(errors) < 1:
        return None
    errors = np.array([e for e in errors if e is not None])
    errors = errors[np.isfinite(errors)]
    if len(errors) < 1:
        return None
    return errors


def safe_rmse(errors:Sequence[float| None] | None | np.ndarray, factor:float = 1)->float|None:
    c_errors = clean_errors(errors=errors)
    if c_errors is None:
        return None
    return np.sqrt(1/c_errors.shape[0] * np.sum((c_errors)**2))*factor


def safe_mae(errors:Sequence[float| None] | None | np.ndarray, factor:float = 1)->float|None:
    c_errors = clean_errors(errors=errors)
    if c_errors is None:
        return None
    return float(np.mean(c_errors))*factor


def safe_median(errors:Sequence[float| None] | None | np.ndarray, factor:float = 1)->float|None:
    c_errors = clean_errors(errors=errors)
    if c_errors is None:
        return None
    return float(np.median(c_errors))*factor


def fmt_rmse(errors:Sequence[float| None] | None | np.ndarray, fmt=".1f", default = "N/A", factor:float = 1.0)->str:
    return format_optional(value=safe_rmse(errors), fmt=fmt, default=default, factor=factor)


def fmt_mae(errors:Sequence[float | None] | None | np.ndarray, fmt=".1f",default = "N/A", factor:float = 1.0, add_se:bool = True)->str:
    c_errors = clean_errors(errors)
    if c_errors is None:
        return default
    c_errors = c_errors*factor
    ret = format_optional(value=safe_mae(c_errors), fmt=fmt, default=default)
    if add_se and ret != default:
        ret += "±"+format_optional(value=np.std(c_errors, ddof = 1)/np.sqrt(c_errors.shape[0]), fmt = fmt, default=default)
    return ret


def fmt_median(errors:Sequence[float | None] | None | np.ndarray, fmt=".1f", default = "N/A", factor:float = 1.0)->str:
    return format_optional(value=safe_median(errors), fmt=fmt, default=default, factor=factor)


def calculate_ray_missalignment_errors(timed_pred_gt_s:list[tuple[int, np.ndarray, np.ndarray]])->list[tuple[int, float, float, float, float]]:
    """
    Calculates the missalignments of the predicted pose in relation to the z-axis of the GD pose
    :param timed_pred_gt_s: List of (i, pred, GT) tuples, with i being the frame idx (unique), and pred, GT in SE(3)
    :return: A list [i, x_error, y_error, x_angle_error, y_angle_error] with x_error and y_error being the translational difference from GT to pred
    and y_ang_error being the angle between the pred. z-axis projected onto the GT x-z plane and the GT z-axis (same for x_ang_error)
    """
    timed_errors = []

    for t, r_t_predh, r_t_gth in timed_pred_gt_s:
        gth_t_predh = np.linalg.inv(r_t_gth) @ r_t_predh

        x_err = gth_t_predh[0,3]
        y_err = gth_t_predh[1,3]

        z_proj_xz = np.array([gth_t_predh[0,2], gth_t_predh[2,2]])
        z_proj_yz = np.array([gth_t_predh[1,2], gth_t_predh[2,2]])

        norm_xz = np.linalg.norm(z_proj_xz)
        norm_yz = np.linalg.norm(z_proj_yz)

        if norm_xz > 1e-10:
            z_proj_xz = z_proj_xz / norm_xz
            x_angle_error = np.arctan2(z_proj_xz[0], z_proj_xz[1])
        else:
            x_angle_error = 0.0

        if norm_yz > 1e-10:
            z_proj_yz = z_proj_yz / norm_yz
            y_angle_error = np.arctan2(z_proj_yz[0], z_proj_yz[1])
        else:
            y_angle_error = 0.0

        timed_errors.append((t, x_err, y_err, x_angle_error, y_angle_error))

    return timed_errors


def calculate_signed_errors(timed_pred_gt_s:list[tuple[int, np.ndarray, np.ndarray]])->np.ndarray:
    """
    Calculates the following signed errors in the ground truth frames:
    XYZ-translational errors
    Rotational error on the xy, yz and xz plane (rotational difference to the frame projected onto the plane)
    :return: Bx6 error array: [[x, y, z , x_rot, y_rot, z_rot], ...]
    """
    if len(timed_pred_gt_s) < 1:
        return np.empty((0,6))

    errors = []
    for _, r_t_predh, r_t_gth in timed_pred_gt_s:
        gth_t_predh = np.linalg.inv(r_t_gth) @ r_t_predh

        # proj xy, yz, xz
        y_proj_yz = gth_t_predh[[1,2], 1]
        x_proj_xz = gth_t_predh[[0,2], 0]
        x_proj_xy = gth_t_predh[[0,1], 0]

        vecs = np.array([y_proj_yz, x_proj_xz, x_proj_xy])
        vecs_n = np.linalg.norm(vecs, axis=-1)

        rot_errors = np.asarray([
            np.arctan2(vecs[i, 1], vecs[i, 0]) if vecs_n[i] > 1e-8 else 0
            for i in range(3)
        ])
        errors.append(np.concatenate([gth_t_predh[:3, 3], rot_errors]))

    return np.asarray(errors)



class PredictionOnDataset:
    def __init__(self,
                 predictor:HeadsetLocalizer,
                 headset_data:HeadsetRecording,
                 number_retry:int = 1,
                 vid_gen:VideoGenerator | None = None,
                 video_save_location:str = "test.mp4",
                 gripping_error:FastRayIntersectionError | None = None,
                 use_tqdm:bool = True
            ):
        """
        Uses the predictor to run predictions on the dataset and gather metrics.
        :param predictor: The predictor that will do the predictions
        :param headset_data: The headset data that provides the prediction frames & maybe labels
        :param number_retry: Max number of retries given to est_base_t_cam
        :param gripping_error: 
        """
        assert isinstance(predictor, HeadsetLocalizer)
        assert isinstance(headset_data, HeadsetRecording)
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
                vid_gen.start_new_frame()

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
        self.signed_errors = calculate_signed_errors(self.comparable_poses)


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
        
        self.rie_s = []
        self.timed_rie_errors = []
        if gripping_error is not None:
            for i, m1, m2 in self.comparable_poses:
                errors = gripping_error.compute_ray_intersection_error_img(
                    base_t_cam1=m1,
                    base_t_cam2=m2,
                    dim=(w, h),
                    size=1,
                    stride=7
                ).reshape(-1)
                errors = errors[np.isfinite(errors)]
                self.rie_s.append(errors)
                self.timed_rie_errors.append((i, np.median(errors)))
            self.rie_s = np.concatenate(self.rie_s) if len(self.rie_s) > 0 else np.array([])
        
        self.median_ray_intersection_error = np.median(self.rie_s) if len(self.rie_s) > 0 else None
        self.mean_ray_intersection_error = np.mean(self.rie_s) if len(self.rie_s) > 0 else None
        self.rmse_ray_intersection_error = safe_rmse(self.rie_s) if len(self.rie_s) > 0 else None

        self.translational_errors = [e for _, e in self.timed_translational_errors]
        self.rotational_errors = [e for _, e in self.timed_rotational_errors]

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

        print(f"\n\nAvg. error: {fmt_mae(self.translational_errors, factor=1000)} mm and {fmt_mae(self.translational_errors, factor=180/np.pi)}°")
        print(f"Median. error: {fmt_median(errors=self.translational_errors, factor=1000)} mm and {fmt_median(self.translational_errors, factor=180/np.pi)}°")
        print(f"ATE RMSE: {fmt_rmse(self.translational_errors, factor=1000)} mm and {fmt_rmse(self.translational_errors, factor=180/np.pi)}°")
        print(f"RTE RMSE: {self.rte_translation_rmse * 1000:.1f} mm and {np.rad2deg(self.rte_rotational_rmse):.1f}°")
        print(f"avg gripping error: {format_optional(self.mean_ray_intersection_error, fmt=".1f", factor=1000)} mm")
        print(f"median gripping error: {format_optional(self.median_ray_intersection_error, fmt=".1f", factor=1000)} mm")


    def plot_signed_error_s(self, axes:list[Axes] | None = None, explain = False):
        
        if axes is None:
            _, axs = plt.subplots(1, 6, figsize=(18, 6), sharey=False)
        else:
            axs = axes

        names = ["TE-X", "TE-Y", "TE-Z", "RE-X", "RE-Y", "RE-Z"]
        if explain:
            names = ["TE-X, >0 → to right", "TE-Y, >0 → to down", "TE-Z, >0 → to close", "RE-X, >0 → looking up", "RE-Y, >0 → looking right", "RE-Z, >0 → slanted head right"]


        for i, (name, ax) in enumerate(zip(names, axs)):
            error_unit = "[mm]" if i < 3 else "[deg]"
            error_multiplyer = 1000 if i < 3 else 180/np.pi

            mult_error = self.signed_errors[:, i]*error_multiplyer

            x_limit = np.max(np.abs(mult_error))*1.1

            sns.histplot(
                data=mult_error, ax=ax, kde=True, alpha=0.5, stat='density'
            )
            ax.set_title(name, fontweight='bold')
            ax.set_xlabel(f'Error {error_unit}')
            ax.set_ylabel('Density' if i == 0 else '', fontsize=9)
            ax.set_xlim(-x_limit, x_limit)
            ax.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=1)

        plt.suptitle('Distribution of the signed errors', fontweight='bold', y=1.02)
        plt.tight_layout()
        plt.show()


    def plot_ray_misalignment(self, ax_t:Axes | None = None, ax_r:Axes | None = None, name:str = ""):
        """
        :param ax_t: The ax to plot the translational errors onto
        :param ax_r: The ax to plot the rotational errors onto
        """
        if ax_t is None or ax_r is None:
            fig, axes = plt.subplots(1,2, figsize = (12, 5))
            fig.suptitle(f"Ray Misalignment Errors {name}", fontsize=14)
            ax_t, ax_r = axes

        timed_errors = calculate_ray_missalignment_errors(timed_pred_gt_s=self.comparable_poses)

        times = [t for t, _, _, _, _ in timed_errors]
        x_t_errs = [x_t_e*1000 for _, x_t_e, _, _, _ in timed_errors]
        y_t_errs = [y_t_e*1000 for _, _, y_t_e, _, _ in timed_errors]
        x_r_errs = [np.rad2deg(x_r_e) for _, _, _, x_r_e, _ in timed_errors]
        y_r_errs = [np.rad2deg(y_r_e) for _, _, _, _, y_r_e in timed_errors]

        df_t = pd.DataFrame({'Time': times, 'X Error': x_t_errs, 'Y Error': y_t_errs})
        df_r = pd.DataFrame({'Time': times, 'X Rotation': x_r_errs, 'Y Rotation': y_r_errs})

        scatter_t = sns.lineplot(data=df_t, x='X Error', y='Y Error', hue='Time', ax=ax_t, marker='s', markersize=5, legend=False, palette="viridis")
        scatter_r = sns.lineplot(data=df_r, x='X Rotation', y='Y Rotation', hue='Time', ax=ax_r, marker='s', markersize=5, legend=False, palette="viridis")

        ax_t.set_title("Translational errors of prediction on GT xy-plane")
        ax_t.set_xlabel("xy-plane x error [mm]")
        ax_t.set_ylabel("xy-plane y error [mm]")
        ax_t.set_aspect('equal')
        t_lim = max(np.max(np.abs(x_t_errs)), np.max(np.abs(y_t_errs)))*1.1
        ax_t.set_xlim(-t_lim, t_lim)
        ax_t.set_ylim(-t_lim, t_lim)
        ax_t.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
        ax_t.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=0.5)

        ax_r.set_title("Rotational errors of prediction on xz and yz plane")
        ax_r.set_xlabel("xz-plane error [deg]")
        ax_r.set_ylabel("yz-plane error [deg]")
        ax_r.set_aspect('equal')
        r_lim = max(np.max(np.abs(x_r_errs)), np.max(np.abs(y_r_errs)))*1.1
        ax_r.set_xlim(-r_lim, r_lim)
        ax_r.set_ylim(-r_lim, r_lim)
        ax_r.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.5)
        ax_r.axvline(x=0, color='black', linestyle='--', alpha=0.3, linewidth=0.5)

    

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
