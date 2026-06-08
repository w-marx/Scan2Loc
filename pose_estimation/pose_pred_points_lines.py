import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.axes import Axes
from dataclasses import dataclass
from numbers import Number
import torch

from shared.assertion_helpers import assert_intrinsic_mat, assert_mxnx3_np_uint8_image_batch, assert_mxnx3_np_uint8_image

from predictor_handling import *
from extractors_and_matchers import *
from geometric_utilities.pnpl_optimizer import *
from geometric_utilities.line_utilities import * 


@dataclass(frozen=True, kw_only=True)
class LineMerging2dConfig:
    """
    Discribes a line merging pass, that merges the lines and then removes
    lines with line_length < min_line_length
    """
    max_angle_diff_deg:float = 3
    max_midpoint_dist_px:float = 3
    max_endpoint_dist_px:float = 20
    min_line_length:float = 20
    use_quick_merge:bool = False

    def __post_init__(self):
        assert isinstance(self.max_angle_diff_deg, Number) and 0 <= self.max_angle_diff_deg <= 180 
        assert isinstance(self.max_angle_diff_deg, Number) and 0 <= self.max_midpoint_dist_px
        assert isinstance(self.max_angle_diff_deg, Number) and 0 <= self.max_endpoint_dist_px
        assert isinstance(self.max_angle_diff_deg, Number) and 0 <= self.min_line_length
        assert isinstance(self.use_quick_merge, bool)

line_merging_2d_config_for_short_lines_quick_merge = LineMerging2dConfig(
    max_angle_diff_deg = 1,
    max_midpoint_dist_px = 2,
    max_endpoint_dist_px = 5,
    min_line_length = 10,
    use_quick_merge = True
)

line_merging_2d_config_for_longer_lines_quick_merge = LineMerging2dConfig(
    max_angle_diff_deg = 2,
    max_midpoint_dist_px = 3,
    max_endpoint_dist_px = 10,
    min_line_length = 40,
    use_quick_merge = True
)

line_merging_2d_config_for_short_lines = LineMerging2dConfig(
    max_angle_diff_deg = 2,
    max_midpoint_dist_px = 3,
    max_endpoint_dist_px = 5,
    min_line_length = 10,
    use_quick_merge = False
)

line_merging_2d_config_for_longer_lines = LineMerging2dConfig(
    max_angle_diff_deg = 3,
    max_midpoint_dist_px = 4,
    max_endpoint_dist_px = 15,
    min_line_length = 40,
    use_quick_merge = False
)

@dataclass(frozen=True, kw_only=True)
class MultiPassLineMergingConfig:
    passes: list[LineMerging2dConfig] = field(
        default_factory=lambda: (
            [
                line_merging_2d_config_for_short_lines_quick_merge,
                line_merging_2d_config_for_longer_lines_quick_merge
            ]
        )
    )

    def __post_init__(self):
        assert len(self.passes) > 0
        assert all([isinstance(obj, LineMerging2dConfig) for obj in self.passes])


@dataclass(frozen=True, kw_only=True)
class LineFitting3dConfig:
    """
    How to fit the 3d points of the xyz-images to a line
    :param use_ransac: If True ransac will be used, else robust PCA (faster but less robust)
    :param ransac_itterations: Number of itterations the ransac algorithm needs
    :param ransac_inlier_distance: Distance in meters to be considered an inlier for the ransac algorithm
    """
    use_ransac:bool = True
    ransac_itterations:int = 100
    ransac_inlier_distance:float = 0.005

    def __post_init__(self):
        assert isinstance(self.use_ransac, bool)
        if self.use_ransac:
            assert isinstance(self.ransac_itterations, Number) and 0 < self.ransac_itterations
            assert isinstance(self.ransac_inlier_distance, Number) and 0 <= self.ransac_inlier_distance




class LinePredictor(PosePredictor):
    def __init__(
            self,
            cam2_intrinsic_mtx:np.ndarray,
            cam1_bgr_images:np.ndarray,
            cam1_xyz_images:np.ndarray,
            time_tracker_init: TimeTracker = TimeTracker(),
            extract_and_match_wrapper_config:ExtractAndMatchWrapperConfig = ExtractAndMatchWrapperConfig(),
            lsd_cleanup_passes_configs:MultiPassLineMergingConfig = MultiPassLineMergingConfig(),
            line_matching_config:LineMatchingConfig = LineMatchingConfig(),
            line_fitting_3d_config:LineFitting3dConfig = LineFitting3dConfig(),
            pnpl_optimisation_conf:PnPLOptimizerConfig = PnPLOptimizerConfig(),
            cam2_lsd_size:None | tuple[int, int] = None,
            debug_visualize_line_cleanup:bool = False,
            debug_visualize_pnpl:bool = False,
        ):
        """
        A predictor that uses points & lines as features

        :param cam2_intrinsic_mtx: The 3x3 intrinsic matrix for camera 2
        :param cam1_bgr_images: BxHxWx3-uint8 array of bgr images for camera 1
        :param cam1_xyz_images: BxHxWx3-float array of xyz-point images for camera 1 in the base ref. frame
        :param time_tracker_init: A timetracker where important steps during the initialization will be registered
        :param extract_and_match_wrapper_config: The configuration for how to extract and match the points
        :param lsd_cleanup_passes_configs: The cleanup passes that will be enacted after LSD on any image
        :param line_matching_config: How to match lines from 2 different images
        :param line_fitting_3d_config: How to fit the 3d lines to the 3d point clouds from cam1_xyz_images
        :param pnpl_optimisation_conf: How to do the PnPL-optimisation
        :param cam2_lsd_size: If not None the cam2 images will be scaled to that resolution before LSD (smaller -> better runtime)
        :param debug_visualize_line_cleanup: If True the line features will be visualised
        :param debug_visualize_pnpl: If True the optimisation by the pnpl-optimisation will be visualized
        """
        super().__init__()
        assert assert_intrinsic_mat(cam2_intrinsic_mtx)
        self.cam2_intrinsic_mtx = cam2_intrinsic_mtx

        assert assert_mxnx3_np_uint8_image_batch(cam1_bgr_images)
        assert cam1_xyz_images.shape == cam1_bgr_images.shape
        self.cam1_xyz_images = cam1_xyz_images

        time_tracker_init.reset_elapsed_time()

        self.extract_and_match_wrapper = ExtractAndMatchWrapper(
            cam2_mtx=cam2_intrinsic_mtx,
            cam1_bgr_images=cam1_bgr_images,
            cam1_xyz_images=cam1_xyz_images,
            config=extract_and_match_wrapper_config
        )
        time_tracker_init.add_time_stamp("ExtractAndMatchWrapper Initialisation")

        self.lsd_cleanup_passes = lsd_cleanup_passes_configs.passes
        self.line_matching_config = line_matching_config

        self.line_fitting_3d_method = (lambda points3d:(
            line_segment_regression_3d_ransaac(
                xyz_points=points3d,
                inlier_distance=line_fitting_3d_config.ransac_inlier_distance,
                itterations=line_fitting_3d_config.ransac_itterations
            )
        )) if line_fitting_3d_config.use_ransac else lambda points3d: robust_pca_2d_3d_points_lineseg_regression(points=points3d)

        self.pnpl_optimisation_conf = pnpl_optimisation_conf


        self.debug_visualize_line_cleanup = debug_visualize_line_cleanup
        self.debug_visualize_pnpl = debug_visualize_pnpl

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.line_seg_detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)

        self.cam1_bgr_images = cam1_bgr_images

        time_tracker_init.reset_elapsed_time()
        self.lines_4_images_cam1 = [
            self._lsd_and_cleanup_on_image(img) for img in cam1_bgr_images
        ]
        time_tracker_init.add_time_stamp("LSD and Cleanup")

        self.cam2_lsd_size = cam2_lsd_size

    @staticmethod
    def get_creation_function(
            cam2_intrinsic_mtx: np.ndarray,
            extract_and_match_wrapper_config: ExtractAndMatchWrapperConfig = ExtractAndMatchWrapperConfig(),
            lsd_cleanup_passes_configs: MultiPassLineMergingConfig = MultiPassLineMergingConfig(),
            line_matching_config: LineMatchingConfig = LineMatchingConfig(),
            line_fitting_3d_config: LineFitting3dConfig = LineFitting3dConfig(),
            pnpl_optimisation_conf: PnPLOptimizerConfig = PnPLOptimizerConfig(),
            cam2_lsd_size: None | tuple[int, int] = None,
            debug_visualize_line_cleanup: bool = False,
            debug_visualize_pnpl: bool = False
    ):
        """
        Returns a function with which a new LinePredictor may be created.
        For parameter info look at `__init__`
        :return: f(robot_env,time_tracker) -> LinePredictor
        """
        creation_function = lambda robot_env, init_tt: LinePredictor(
            cam2_intrinsic_mtx=cam2_intrinsic_mtx,
            cam1_bgr_images=robot_env.robot_bgr_images,
            cam1_xyz_images=robot_env.robot_xyz_images,
            extract_and_match_wrapper_config=extract_and_match_wrapper_config,
            lsd_cleanup_passes_configs = lsd_cleanup_passes_configs,
            line_matching_config = line_matching_config,
            line_fitting_3d_config = line_fitting_3d_config,
            pnpl_optimisation_conf = pnpl_optimisation_conf,
            cam2_lsd_size = cam2_lsd_size,
            debug_visualize_line_cleanup = debug_visualize_line_cleanup,
            debug_visualize_pnpl = debug_visualize_pnpl,
            time_tracker_init=init_tt
        )
        return creation_function


    @staticmethod
    def visualize_line_cleanup(
            lines_before:np.ndarray,
            lines_after:np.ndarray,
            background_image:np.ndarray,
            ax_before: Axes | None = None,
            ax_after: Axes | None = None,
    )->None:
        """
        Visualises how the lines change through cleanup
        :param lines_before: Nx4 array of line segments of the style [[x0, y0, x1, y1], ...]
        :param lines_before: Mx4 array of line segments of the style [[x0, y0, x1, y1], ...]
        :param background_image_bgr: HxWx3 BGR image as numpy array or HxW Greyscale image
        :param ax_before: An matplotlib axis on which before will be plotted (if None it will be created and the plot shown)
        :param ax_after: Same as ax_before for after
        """
        has_to_plot = ax_before is None or ax_after is None
        if has_to_plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 10))
            ax_before = axes[0]
            ax_after = axes[1]

        if background_image.ndim == 3:
            background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2RGB)

        ax_before.imshow(background_image)
        ax_after.imshow(background_image)

        lines_xy_raw = [((line[0], line[1]), (line[2], line[3])) for line in lines_before]
        lc1_raw = LineCollection(lines_xy_raw, linewidths=2, alpha=0.8, color = plt.cm.jet(np.linspace(0, 1, lines_before.shape[0])))
        ax_before.add_collection(lc1_raw)
        ax_before.set_title("Lines before cleanup")

        line_colors_processed = plt.cm.jet(np.linspace(0, 1, lines_after.shape[0]))
        lines_xy_processed = [((line[0], line[1]), (line[2], line[3])) for line in lines_after]
        lc1_processed = LineCollection(lines_xy_processed, linewidths=2, alpha=0.8, color = line_colors_processed)
        ax_before.add_collection(lc1_processed)
        ax_before.set_title("Lines after cleanup")

        if has_to_plot:
            plt.show()


    def visualize_features_2d(
            self, 
            img1:np.ndarray, 
            img2:np.ndarray, 
            lines1_raw:np.ndarray, 
            lines2_raw:np.ndarray, 
            lines1_processed:np.ndarray, 
            lines2_processed:np.ndarray, 
            points1:np.ndarray, 
            points2:np.ndarray,
        ):
        #TODO reuse for video generation
        """
        :param img1: HxWx3 BGR image as numpy array
        :param img2: HxWx3 RGB image as numpy array
        :param lines1: Nx4 array of line segments
        :param lines2: Nx4 array of line segments
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Display the raw lines
        lines_xy1_raw = [((line[0], line[1]), (line[2], line[3])) for line in lines1_raw]
        lc1_raw = LineCollection(lines_xy1_raw, linewidths=2, alpha=0.8, color = plt.cm.jet(np.linspace(0, 1, lines1_raw.shape[0])))
        axes[0, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[0, 0].add_collection(lc1_raw)
        axes[0, 0].set_title("Raw lines cam1")

        lines_xy2_raw = [((line[0], line[1]), (line[2], line[3])) for line in lines2_raw]
        lc2_raw = LineCollection(lines_xy2_raw, linewidths=2, alpha=0.8, color = plt.cm.jet(np.linspace(0, 1, lines2_raw.shape[0])))
        axes[0, 1].imshow(img2)
        axes[0, 1].add_collection(lc2_raw)
        axes[0, 1].set_title("Raw lines cam1")

        # Display the processed lines
        point_colors = plt.cm.jet(np.linspace(0, 1, points1.shape[0]))
        line_colors = plt.cm.jet(np.linspace(0, 1, lines1_processed.shape[0]))

        lines_xy1 = [((line[0], line[1]), (line[2], line[3])) for line in lines1_processed]
        lc1 = LineCollection(lines_xy1, linewidths=2, alpha=0.8, color = line_colors)
        axes[1, 0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        axes[1, 0].add_collection(lc1)
        axes[1, 0].set_title("Processed features cam1")
        axes[1, 0].scatter(points1[:, 0], points1[:, 1], s = 2, color = point_colors, alpha = 0.8)


        lines_xy2 = [((line[0], line[1]), (line[2], line[3])) for line in lines2_processed]
        lc2 = LineCollection(lines_xy2, linewidths=2, alpha=0.8, color = line_colors)
        axes[1, 1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        axes[1, 1].add_collection(lc2)
        axes[1, 1].set_title("Processed features cam2")
        axes[1, 1].scatter(points2[:, 0], points2[:, 1], s = 2, color = point_colors, alpha = 0.8)

        plt.show()


    def _est_base_t_cam2_4_idx(
            self,
            cam2_rgb_image: np.ndarray,
            idx:int,
            lines_img2:np.ndarray,
            time_tracker:TimeTracker = TimeTracker()
        ) -> np.ndarray | None:
        """
        Estimates base_t_cam2 for a given cam1-image index
        :param cam2_rgb_image: HxWx3 BGR image as numpy array
        :param idx: The index of the cam1 datapoint to use as reference
        :param lines_img2: Nx4 array of line segments in the cam2-view
        :param time_tracker: a time tracker where subcomponent times will be tracked
        """
        time_tracker.reset_elapsed_time()
        base_t_cam_and_points = self.extract_and_match_wrapper.est_base_t_cam2_and_points(
            idx=idx, cam2_rgb_image=cam2_rgb_image
        )
        time_tracker.add_time_stamp("point feature pose pred")

        if base_t_cam_and_points is None:
            return None
        
        
        base_t_cam_pnp, image_points_cam1, image_points_cam2, world_obj_points, inliers = base_t_cam_and_points
        lines_img1 = self.lines_4_images_cam1[idx]

        line_pairs = match_2d_line_segments(
            lines_img1=lines_img1, 
            lines_img2=lines_img2,
            points_img1=image_points_cam1,
            points_img2=image_points_cam2,
            line_matching_config=self.line_matching_config
        )
        time_tracker.add_time_stamp("Match 2d line segments")


        matched_lines_2d = []
        matched_lines_3d = []
        
        xyz_points_4_lines = [line_seg_2d_to_3d_points(line_pair[0], self.cam1_xyz_images[idx]) for line_pair in line_pairs]

        for i, line_pair in enumerate(line_pairs):
            line_segment_3d = self.line_fitting_3d_method(xyz_points_4_lines[i])
            if line_segment_3d is not None:
                matched_lines_3d.append(line_segment_3d)
                matched_lines_2d.append(line_pair[1])
        matched_lines_2d = np.array(matched_lines_2d) if len(matched_lines_2d) > 0 else np.empty((0,4))
        matched_lines_3d = np.array(matched_lines_3d) if len(matched_lines_3d) > 0 else np.empty((0,6))

        time_tracker.add_time_stamp("Line 2d -> 3d transformation")
        
        time_tracker.reset_elapsed_time()

        cam2_t_base_bundle_adjustment = optimize_pnpl(
            initial_cam_t_base=np.linalg.inv(base_t_cam_pnp).copy(),
            points_3d=world_obj_points[inliers].copy(),
            points_2d=image_points_cam2[inliers].copy(),
            intrinsic_cam_mat=self.cam2_intrinsic_mtx.copy(),
            lines_2d = matched_lines_2d,
            lines_3d = matched_lines_3d,
            config=self.pnpl_optimisation_conf,
            visualize_result= cam2_rgb_image if self.debug_visualize_pnpl else None
        )

        time_tracker.add_time_stamp("PnL Optimisation")
        return np.linalg.inv(cam2_t_base_bundle_adjustment)



    def _cleanup_lines(self, lines:np.ndarray)->np.ndarray:
        """
        Takes the lines and cleans them up according to the cleanup-config of the instance
        :param lines: Bx4 array of lines of the style: [[x0, y0, x1, y2], ... ]
        :return: B'x4 array of lines of the style: [[x0, y0, x1, y2], ... ], with B' <= B
        """
        lines = lines
        for ref_conf in self.lsd_cleanup_passes:
            lines = remove_short_2d_line_segments(
                merge_close_line_segments(
                    lines,ref_conf.max_angle_diff_deg,ref_conf.max_midpoint_dist_px, ref_conf.max_endpoint_dist_px, ref_conf.use_quick_merge
                ),
            min_line_length_px= ref_conf.min_line_length)
        return lines
    

    def _lsd_and_cleanup_on_image(self, bgr_image:np.ndarray, lsd_at_size: None | tuple[int, int] = None)->np.ndarray:
        """
        Runs LSD on an image that can be scaled down beforehand, then cleans those lines up and scales them back
        :param bgr_image: The HxWx3-uint8 BGR image to be done lsd upon
        :param lsd_at_size: None or a tuple: (height, width) in px
        :return Nx4 line array of the format: [[x1, y1, x2, y2], ...]
        """
        assert assert_mxnx3_np_uint8_image(bgr_image)

        cam2_grey_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        if lsd_at_size is None:
            raw_lines = self.line_seg_detector.detect(cam2_grey_img)[0].squeeze(1)
            clean_lines = self._cleanup_lines(raw_lines)
            if self.debug_visualize_line_cleanup:
                self.visualize_line_cleanup(lines_before=raw_lines,lines_after=clean_lines,background_image=cam2_grey_img)
            return clean_lines
        
        # In case of scaling:
        h_orig, w_orig = bgr_image.shape[:2]
        h_scaled, w_scaled = lsd_at_size

        cam2_image_scaled = cv2.resize(cam2_grey_img, (w_scaled, h_scaled), interpolation = cv2.INTER_AREA)
        lines_scaled_raw = self.line_seg_detector.detect(cam2_image_scaled)[0].squeeze(1)
        lines_scaled = self._cleanup_lines(lines_scaled_raw)

        if self.debug_visualize_line_cleanup:
            self.visualize_line_cleanup(
                lines_before=lines_scaled_raw,
                lines_after=lines_scaled,
                background_image=cam2_image_scaled
            )

        if lines_scaled.size > 0:
            lines_scaled[:, [0,2]] *= w_orig/w_scaled
            lines_scaled[:, [1,3]] *= h_orig/h_scaled

        return lines_scaled

    

    def est_base_t_cam2(self,cam2_bgr_image: np.ndarray, number_retry:int = 1, time_tracker:TimeTracker = TimeTracker()) -> np.ndarray | None:
        """
        Estimates the hom. transformation: baseT_cam2 based on point and line features
        :param cam2_bgr_image: The camera 2 image (HxWx3-uint8 array)
        :param number_retry: With how many different cam1 images the prediction may be tried (upper bound)
        :param time_tracker: A time tracker where timestamps for the different subcomponents will be added.
        :return: The 4x4 Pose in SE3 if prediction was successful else None
        """
        assert assert_mxnx3_np_uint8_image(cam2_bgr_image)
        assert number_retry > 0

        number_tries = 0
        est_base_t_cam = None
        cam2_rgb_image = cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)


        time_tracker.reset_elapsed_time()
        lines_img2 = self._lsd_and_cleanup_on_image(cam2_bgr_image, lsd_at_size=self.cam2_lsd_size)
        time_tracker.add_time_stamp("Image 2 LSD + cleanup")

        while est_base_t_cam is None and number_tries < number_retry:
            idx = self.extract_and_match_wrapper.sheduler.get_best()
            est_base_t_cam = self._est_base_t_cam2_4_idx(
                idx=idx,
                lines_img2 = lines_img2,
                cam2_rgb_image = cam2_rgb_image,
            )
            self.extract_and_match_wrapper.sheduler.adjust(idx, est_base_t_cam is not None)
            number_tries += 1
        return est_base_t_cam
    
    def update_pose(self,cam2_bgr_image: np.ndarray, rough_base_t_cam2:np.ndarray, time_tracker:TimeTracker) -> np.ndarray | None:
        """
        Acts exactly the same as est_base_t_cam2 with this predictor
        :param cam2_bgr_image: HxWx3 bgr image
        :param rough_base_t_cam2: A rough base_t_cam2 estimate.
        :param time_tracker: a time-tracker object, that will be used by the Pose Predictor to note the runtimes
        :return: 4x4 Pose in SE3 if prediction was successful else None
        """
        return self.est_base_t_cam2(cam2_bgr_image=cam2_bgr_image, number_retry=1, time_tracker=time_tracker)





if __name__ == "__main__":
    robot_data = RobotEnvironment.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/pose_estimation/out_data_re")
    headset_data = HeadsetData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/pose_estimation/out_data_he")
    
    
    predictor = LinePredictor(
        cam2_intrinsic_mtx=headset_data.intrinsic_cam_mtx,
        cam1_bgr_images=robot_data.robot_bgr_images,
        cam1_xyz_images=robot_data.robot_xyz_images,
        extract_and_match_wrapper_config=ExtractAndMatchWrapperConfig(
            extract_and_match=ExtractAndLightGlue(
                extractor="SuperPoint"
            ),
            ransac_config=pose_estimation_ransaac_config_less_precise,
        ),
        line_fitting_3d_config=LineFitting3dConfig(use_ransac=False),
        pnpl_optimisation_conf=PnPLOptimizerConfig(lm_max_steps=10000,line_relevance=1.0),
        cam2_lsd_size=(514,514),
        debug_visualize_2d=True, 
        debug_visualize_pnpl=False
    )

    tt1 = TimeTracker()
    tt2 = TimeTracker()
    grader = OnePredictorRecordingGrader(
        predictor=predictor, 
        headset_data=headset_data,
        prediction_time_tracker=tt1,
        subcomponent_time_tracker=tt2
    )
    
    print(f"avg rot error: {np.round(np.rad2deg(grader.avg_rotational_error()), 2)} degrees")
    print(f"avg translational error: {np.round(grader.avg_translational_error()*1000, 1)} mm")
    print(f"median rot error: {np.round(np.rad2deg(grader.median_rotational_error()), 2)} degrees")
    print(f"median translational error: {np.round(grader.median_translational_error()*1000, 1)} mm")
    print(f"sucess_ratio: {np.round(grader.success_ratio(),2)}")

    grader.visualize_predictions(robot_env=robot_data)

    print(f"tt1:")
    tt1.print_report()
    print(f"\n tt2:")
    tt2.print_report()

    predictor.extract_and_match_wrapper.print_used_augmentations()
    