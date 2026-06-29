import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number
import torch
from typing import Literal

from shared.assertion_helpers import assert_intrinsic_mat, assert_mxnx3_np_uint8_image, assert_bgr_xyz_image_pair_batch, assert_homogeneous_mat

from ..utilities.slam2mp4 import FeatureDrawing
from ..predictor_handling.pose_predictor import PosePredictor
from ..pnp.extract_and_match_wrapper import ExtractAndMatchWrapperConfig, ExtractAndMatchWrapper
from ..utilities.time_tracker import TimeTracker, TimeLabels
from ..utilities.point_utilities import project_visible_points

from .pnpl_optimizer import optimize_pnpl, PnPLOptimizerConfig
from .line_utilities import (
    assert_nd_line_batch, project_point_onto_line_slow, LineMatchingConfig, match_2d_line_segments,
    line_segment_regression_3d_ransaac, robust_pca_2d_3d_points_lineseg_regression, line_seg_2d_to_3d_points,
    pca_2d_3d_points_lineseg_regression
)
from .line_generator import LineGenerator


@dataclass(frozen=True, kw_only=True)
class LineFitting3dConfig:
    """
    How to fit the 3d points of the xyz-images to a line
    :param use_ransac: If True ransac will be used, else robust PCA (faster but less robust)
    :param ransac_iterations: Number of iterations the ransac algorithm needs
    :param ransac_inlier_distance: Distance in meters to be considered an inlier for the ransac algorithm
    """
    fitting_algorithm:Literal['ransac', 'pca', 'robust_pca'] = 'ransac'
    ransac_iterations:int = 100
    ransac_inlier_distance:float = 0.005

    pca_iterations:int = 3
    pca_inlier_quantile:float = 0.95
    

    def __post_init__(self):
        if self.fitting_algorithm == 'ransac':
            assert isinstance(self.ransac_iterations, Number) and 0 < self.ransac_iterations
            assert isinstance(self.ransac_inlier_distance, Number) and 0 <= self.ransac_inlier_distance

        if self.fitting_algorithm == 'robust_pca':
            assert isinstance(self.pca_iterations, Number) and 0 < self.pca_iterations
            assert isinstance(self.pca_inlier_quantile, Number) and 0 <= self.pca_inlier_quantile


def visualize_features_2d(
        fd:FeatureDrawing,
        obs_lines_matched_2d:np.ndarray, 
        proj_lines_matched_3d:np.ndarray,
        cam_t_base:np.ndarray,
        intrinsic_mat:np.ndarray
    ):
    """
    Paints the lines onto the axis of the fd
    :param fd: The feature drawing object to draw upon using its style
    :param obs_lines_matched_2d: The observed matched lines (Format: [[x0, y0, x1, y1], ...])
    :param proj_lines_matched_3d: The matched 3d lines (Format: [[x0, y0, z0, x1, y1, z1], ...])
    :param cam_t_base: A 4x4 SE3 matrix: cam -> base
    :param intrinsic_mat: The 3x3 intrinsic camera matrix
    """
    assert assert_nd_line_batch(obs_lines_matched_2d, dim=2)
    assert assert_nd_line_batch(proj_lines_matched_3d, dim=3)
    assert assert_homogeneous_mat(cam_t_base, size = 4)
    assert assert_intrinsic_mat(intrinsic_mat)

    n_matched_lines = obs_lines_matched_2d.shape[0]
    colors_lines = plt.cm.jet(np.linspace(0,1, n_matched_lines))

    for i,(x1, y1, x2, y2) in enumerate(obs_lines_matched_2d):
        fd.ax.axline((x1, y1), (x2, y2), color=colors_lines[i], linestyle=fd.sc.obs_line_style, linewidth=1)
        fd.ax.plot([x1, x2], [y1, y2], color=colors_lines[i], linewidth=1)

    for i, (line_3d, obs_line) in enumerate(zip(proj_lines_matched_3d, obs_lines_matched_2d)):
        projected_points = project_visible_points(
            base_points=line_3d.reshape(-1, 3),
            cam_t_base=cam_t_base,
            intrinsic_mat=intrinsic_mat,
        )
        fd.ax.scatter(
            projected_points[:,0],
            projected_points[:,1],
            color=colors_lines[i],
            s=fd.sc.point_size, alpha=fd.sc.point_alpha, marker = fd.sc.proj_point_style
        )

        x1, y1 = projected_points[0]
        x2, y2 = projected_points[1]
        ox0, oy0, ox1, oy1 = obs_line

        px0, py0 = project_point_onto_line_slow(x1, y1, ox0, oy0, ox1, oy1)
        px1, py1 = project_point_onto_line_slow(x2, y2, ox0, oy0, ox1, oy1)

        fd.ax.quiver(
            [x1, x2],
            [y1, y2],
            [px0 - x1, px1 - x2],
            [py0 - y1, py1 - y2],
            angles='xy', scale_units='xy', scale=1,
            color=colors_lines[i], alpha=fd.sc.arrow_alpha, width=0.005
        )


class LinePredictor(PosePredictor):
    def __init__(
            self,
            cam2_intrinsic_mtx:np.ndarray,
            cam1_bgr_images:np.ndarray,
            cam1_xyz_images:np.ndarray,
            time_tracker_init: TimeTracker = TimeTracker(),
            extract_and_match_wrapper_config:ExtractAndMatchWrapperConfig = ExtractAndMatchWrapperConfig(),
            cam1_line_generator:LineGenerator = LineGenerator(),
            cam2_line_generator:LineGenerator | None = None,
            line_matching_config:LineMatchingConfig = LineMatchingConfig(),
            line_fitting_3d_config:LineFitting3dConfig = LineFitting3dConfig(),
            pnpl_optimisation_conf:PnPLOptimizerConfig = PnPLOptimizerConfig(),
            debug_visualize_pnpl:bool = False,
        ):
        """
        A predictor that uses points & lines as features
        :param cam2_intrinsic_mtx: The 3x3 intrinsic matrix for camera 2.
        :param cam1_bgr_images: BxHxWx3-uint8 array of bgr images for camera 1.
        :param cam1_xyz_images: BxHxWx3-float array of xyz-point images for camera 1 in the base ref. frame.
        :param time_tracker_init: A timetracker where important steps during the initialization will be registered.
        :param extract_and_match_wrapper_config: The configuration for how to extract and match the points.
        :param cam1_line_generator: Line generator to extract lines from the cam1 images
        :param cam2_line_generator: A line generator for the cam2 images or None (if none will use the cam1_line_generator for cam 2)
        :param line_matching_config: How to match lines from 2 different images.
        :param line_fitting_3d_config: How to fit the 3d lines to the 3d point clouds from cam1_xyz_images.
        :param pnpl_optimisation_conf: How to do the PnPL-optimization.
        :param debug_visualize_pnpl: If True the optimization by the pnpl-optimization will be visualized.
        """
        super().__init__()
        time_tracker_init.reset_elapsed_time()
        assert assert_intrinsic_mat(cam2_intrinsic_mtx)
        self.cam2_intrinsic_mtx = cam2_intrinsic_mtx

        assert assert_bgr_xyz_image_pair_batch(bgr_images=cam1_bgr_images, xyz_images=cam1_xyz_images)
        self.cam1_xyz_images = cam1_xyz_images
        self.cam1_bgr_images = cam1_bgr_images

        self.cam2_line_generator = None
        if cam2_line_generator is None:
            self.cam2_line_generator = cam1_line_generator


        self.line_matching_config = line_matching_config

        self.line_fitting_3d_method = None

        if line_fitting_3d_config.fitting_algorithm == 'ransac':
            self.line_fitting_3d_method = lambda points3d: line_segment_regression_3d_ransaac(
                xyz_points=points3d,
                inlier_distance=line_fitting_3d_config.ransac_inlier_distance,
                itterations=line_fitting_3d_config.ransac_iterations
            )
        elif line_fitting_3d_config.fitting_algorithm == 'robust_pca':
            self.line_fitting_3d_method = lambda points3d: robust_pca_2d_3d_points_lineseg_regression(
                points=points3d,
                line_distance_quantile=line_fitting_3d_config.pca_inlier_quantile,
                itterations=line_fitting_3d_config.pca_iterations
            )
        else:
            self.line_fitting_3d_method = lambda points3d: pca_2d_3d_points_lineseg_regression(points3d) 


        self.pnpl_optimisation_conf = pnpl_optimisation_conf
        self.debug_visualize_pnpl = debug_visualize_pnpl
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        time_tracker_init.add_time_stamp(TimeLabels.SIMPLE_ATTRIBUTE_INIT)


        self._extract_and_match_wrapper = ExtractAndMatchWrapper(
            cam2_mtx=cam2_intrinsic_mtx,
            cam1_bgr_images=cam1_bgr_images,
            cam1_xyz_images=cam1_xyz_images,
            config=extract_and_match_wrapper_config
        )
        time_tracker_init.add_time_stamp(TimeLabels.EXTRACT_AND_MATCH_WRAPPER_INIT)

        self.lines_4_images_cam1 = [
            cam1_line_generator.get_lines(img) for img in cam1_bgr_images
        ]
        time_tracker_init.add_time_stamp(TimeLabels.LSD_AND_CLEANUP)



    @staticmethod
    def get_creation_function(
            cam2_intrinsic_mtx: np.ndarray,
            extract_and_match_wrapper_config: ExtractAndMatchWrapperConfig = ExtractAndMatchWrapperConfig(),
            cam1_line_generator:LineGenerator = LineGenerator(),
            cam2_line_generator:LineGenerator | None = None,
            line_matching_config:LineMatchingConfig = LineMatchingConfig(),
            line_fitting_3d_config:LineFitting3dConfig = LineFitting3dConfig(),
            pnpl_optimisation_conf:PnPLOptimizerConfig = PnPLOptimizerConfig(),
            debug_visualize_pnpl:bool = False,
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
            cam1_line_generator=cam1_line_generator,
            cam2_line_generator=cam2_line_generator,
            line_matching_config = line_matching_config,
            line_fitting_3d_config = line_fitting_3d_config,
            pnpl_optimisation_conf = pnpl_optimisation_conf,
            debug_visualize_pnpl = debug_visualize_pnpl,
            time_tracker_init=init_tt
        )
        return creation_function


    def _est_base_t_cam2_4_idx(
            self,
            cam2_rgb_image: np.ndarray,
            idx:int,
            lines_img2:np.ndarray,
            time_tracker:TimeTracker = TimeTracker(),
            fd:FeatureDrawing | None = None
        ) -> np.ndarray | None:
        """
        Estimates base_t_cam2 for a given cam1-image index
        :param cam2_rgb_image: HxWx3 BGR image as numpy array
        :param idx: The index of the cam1 datapoint to use as reference
        :param lines_img2: Nx4 array of line segments in the cam2-view
        :param time_tracker: a time tracker where subcomponent times will be tracked
        """
        time_tracker.reset_elapsed_time()
        base_t_cam_and_points = self._extract_and_match_wrapper.est_base_t_cam2_and_points(
            idx=idx, cam2_rgb_image=cam2_rgb_image, fd=fd
        )
        time_tracker.add_time_stamp(TimeLabels.EXTRACT_AND_MATCH_WRAPPER_CALL)

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
        time_tracker.add_time_stamp(TimeLabels.LINE_MATCHING)

        if line_pairs.shape[0] < 1:
            return base_t_cam_pnp


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

        time_tracker.add_time_stamp(TimeLabels.LINE_2D_2_3D)

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

        time_tracker.add_time_stamp(TimeLabels.PNL_OPTIMIZATION)

        if fd is not None:
            visualize_features_2d(
                fd = fd,
                obs_lines_matched_2d=matched_lines_2d,
                proj_lines_matched_3d=matched_lines_3d,
                cam_t_base=cam2_t_base_bundle_adjustment if cam2_t_base_bundle_adjustment is not None else base_t_cam_pnp,
                intrinsic_mat= self.cam2_intrinsic_mtx
            )

        return np.linalg.inv(cam2_t_base_bundle_adjustment)
    

    def est_base_t_cam2(
            self,
            cam2_bgr_image: np.ndarray, 
            number_retry:int = 1, 
            time_tracker:TimeTracker = TimeTracker(),
            fd:FeatureDrawing | None = None
        ) -> np.ndarray | None:
        """
        Estimates the hom. transformation: baseT_cam2 based on point and line features
        :param cam2_bgr_image: The camera 2 image (HxWx3-uint8 array)
        :param number_retry: With how many different cam1 images the prediction may be tried (upper bound)
        :param time_tracker: A time tracker where timestamps for the different subcomponents will be added.
        :return: The 4x4 Pose in SE3 if prediction was successful else None
        """
        assert assert_mxnx3_np_uint8_image(cam2_bgr_image)
        assert number_retry > 0, f"Number retry cant be smaller then 1, is: {number_retry}"

        number_tries = 0
        est_base_t_cam = None
        cam2_rgb_image = cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)

        time_tracker.reset_elapsed_time()
        lines_img2 = self.cam2_line_generator.get_lines(cam2_bgr_image)
        time_tracker.add_time_stamp(TimeLabels.LSD_AND_CLEANUP)

        while est_base_t_cam is None and number_tries < number_retry:
            idx = self._extract_and_match_wrapper.sheduler.get_best()
            est_base_t_cam = self._est_base_t_cam2_4_idx(
                idx=idx,
                lines_img2 = lines_img2,
                cam2_rgb_image = cam2_rgb_image,
                time_tracker = time_tracker,
                fd = fd
            )
            self._extract_and_match_wrapper.sheduler.adjust(idx, est_base_t_cam is not None)
            number_tries += 1
        return est_base_t_cam
    
    @property
    def extract_and_match_wrapper(self)->ExtractAndMatchWrapper | None:
        return self._extract_and_match_wrapper