import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from numbers import Number
import torch
from typing import Literal, Any, Callable
from matplotlib.axes import Axes

from shared.assertion_helpers import assert_intrinsic_mat, assert_mxnx3_np_uint8_image, assert_bgr_xyz_image_pair_batch, assert_homogeneous_mat

from ..utilities.slam2mp4 import FeatureDrawing
from ..localizer_handling.headset_localizer import HeadsetLocalizer
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

    pca_iterations:int = 30
    pca_inlier_quantile:float = 0.95
    min_number_points:int = 10


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

def draw_matched_lines(
        bgr_img1:np.ndarray,
        lines_img1:np.ndarray,
        bgr_img2:np.ndarray,
        lines_img2:np.ndarray,
        lines_img1_unmatched:np.ndarray | None = None,
        lines_img2_unmatched:np.ndarray | None = None,
        axes:list[Axes] | None = None, 
    ):
        
        n_matched_lines = lines_img1.shape[0]

        if axes is None:
            fig, axs = plt.subplots(1,2, figsize = (12, 6))
        else:
            axs = axes

        axs[0].imshow(bgr_img1)
        axs[1].imshow(bgr_img2)

        h1, w1 = bgr_img1.shape[:2]
        h2, w2 = bgr_img2.shape[:2]
        axs[0].set_xlim(0, w1)
        axs[0].set_ylim(h1, 0)
        axs[1].set_xlim(0, w2)
        axs[1].set_ylim(h2, 0)

        if lines_img1_unmatched is not None:
            for i,(x1, y1, x2, y2) in enumerate(lines_img1_unmatched):
                axs[0].plot([x1, x2], [y1, y2], color="gray", linewidth=1)

        if lines_img2_unmatched is not None:
            for i,(x1, y1, x2, y2) in enumerate(lines_img2_unmatched):
                axs[1].plot([x1, x2], [y1, y2], color="gray", linewidth=1)

        colors_lines = plt.cm.jet(np.linspace(0,1, n_matched_lines))

        for i,(x1, y1, x2, y2) in enumerate(lines_img1):
            axs[0].plot([x1, x2], [y1, y2], color=colors_lines[i], linewidth=2)

        for i,(x1, y1, x2, y2) in enumerate(lines_img2):
            axs[1].plot([x1, x2], [y1, y2], color=colors_lines[i], linewidth=2)





def visualize_lines_3d(
        points:np.ndarray,
        lines:np.ndarray
):
    """
    :param points: Nx3 array
    :param lines: Nx6 array
    """
    import open3d as o3d
    points = points[~np.isnan(points).any(axis=1)]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color([0.3, 0.3, 0.3])
    

    lines = lines[~np.isnan(lines).any(axis=1)]

    print(f"lines: {lines.shape}")

    line_points = []
    line_indices = []
    
    for i, line in enumerate(lines):
        start_point = line[:3]
        end_point = line[3:]
        line_points.append(start_point)
        line_points.append(end_point)    
        line_indices.append([2*i, 2*i + 1])
    
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(np.array(line_points))
    line_set.lines = o3d.utility.Vector2iVector(np.array(line_indices))
    line_set.paint_uniform_color([1.0, 0.0, 0.0]) 

    print("=== LineSet Information ===")
    print(f"Number of points: {len(line_set.points)}")
    print(f"Number of lines: {len(line_set.lines)}")
    print(f"Has colors: {line_set.has_colors()}")
    if line_set.has_colors():
        print(f"Colors shape: {np.array(line_set.colors).shape}")
    
    o3d.visualization.draw_geometries([pcd, line_set])


def wrap_filter_points_with_threshold(points3d:np.ndarray, method:Callable[[np.ndarray], np.ndarray | None], min_points:int = 10)-> None | np.ndarray:
    valid = np.isfinite(points3d).all(axis=1)
    filtered_points = points3d[valid]
    if len(filtered_points) < min_points:
        return None
    return method(filtered_points)



class PnPLLocalizer(HeadsetLocalizer):
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
            debug_visualize_matching:bool = False,
            debug_visualize_3d:bool = False
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

        self.cam2_line_generator = cam2_line_generator
        if cam2_line_generator is None:
            self.cam2_line_generator = cam1_line_generator


        self.line_matching_config = line_matching_config

        if line_fitting_3d_config.fitting_algorithm == 'ransac':
            self.orig_method = lambda points3d: line_segment_regression_3d_ransaac(
                xyz_points=points3d,
                inlier_distance=line_fitting_3d_config.ransac_inlier_distance,
                itterations=line_fitting_3d_config.ransac_iterations
            )
        elif line_fitting_3d_config.fitting_algorithm == 'robust_pca':
            self.orig_method = lambda points3d: robust_pca_2d_3d_points_lineseg_regression(
                points=points3d,
                line_distance_quantile=line_fitting_3d_config.pca_inlier_quantile,
                itterations=line_fitting_3d_config.pca_iterations
            )
        else:
            self.orig_method = lambda points3d: pca_2d_3d_points_lineseg_regression(points3d) 

        self.line_fitting_3d_method = lambda points3d: (
            wrap_filter_points_with_threshold(
                points3d=points3d,
                method=self.orig_method,
                min_points=line_fitting_3d_config.min_number_points
            )
        )

        self.pnpl_optimisation_conf = pnpl_optimisation_conf
        self.debug_visualize_pnpl = debug_visualize_pnpl
        self.debug_visualize_matching = debug_visualize_matching
        self.debug_visualize_3d = debug_visualize_3d
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        time_tracker_init.add_time_stamp(TimeLabels.SIMPLE_ATTRIBUTE_INIT)


        self._extract_and_match_wrapper = ExtractAndMatchWrapper(
            cam2_mtx=cam2_intrinsic_mtx,
            cam1_bgr_images=cam1_bgr_images,
            cam1_xyz_images=cam1_xyz_images,
            config=extract_and_match_wrapper_config
        )
        time_tracker_init.add_time_stamp(TimeLabels.EXTRACT_AND_MATCH_WRAPPER_INIT)

        self.lines_4_images_cam1 = []
        self.lines3d = []

        for bgr_img, xyz_img in zip(cam1_bgr_images, cam1_xyz_images):
            lines2d = cam1_line_generator.get_lines(bgr_img)
            lines3d = [self.line_fitting_3d_method(line_seg_2d_to_3d_points(line2d, xyz_img)) for line2d in lines2d]

            self.lines_4_images_cam1.append(np.asarray([l2d for l2d, l3d in zip(lines2d, lines3d) if l3d is not None]))
            self.lines3d.append(np.asarray([l3d for l3d in lines3d if l3d is not None]))

        #print(f"lines 3d: {len(self.lines3d)} x {[x.shape for x in self.lines3d]}")

        time_tracker_init.add_time_stamp(TimeLabels.LSD_CLEANUP_2_3D)


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
            debug_visualize_matching:bool = False,
            debug_visualize_3d:bool = False
    ):
        """
        Returns a function with which a new LinePredictor may be created.
        For parameter info look at `__init__`
        :return: f(robot_env,time_tracker) -> LinePredictor
        """
        creation_function = lambda robot_env, init_tt: PnPLLocalizer(
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
            debug_visualize_matching = debug_visualize_matching,
            time_tracker_init=init_tt,
            debug_visualize_3d = debug_visualize_3d
        )
        return creation_function


    def _est_base_t_cam2_4_idx(
            self,
            cam2_rgb_image_features: list[Any],
            backward_transformations_names: list[tuple[str, Callable[[np.ndarray], np.ndarray]]],
            cam2_augmented_images: np.ndarray,
            cam2_rgb_image:np.ndarray,
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
            idx=idx, 
            cam2_rgb_image_features=cam2_rgb_image_features, 
            backward_transformations_names = backward_transformations_names,
            augmented_images=cam2_augmented_images,
        )


        if fd is not None:
            fd.visualize_localizer(
                robot_img_rgb=cv2.cvtColor(self._extract_and_match_wrapper.cam1_bgr_images_for_vis[idx], code=cv2.COLOR_BGR2RGB),
                headset_img_rgb=cam2_rgb_image
            )


        time_tracker.add_time_stamp(TimeLabels.EXTRACT_AND_MATCH_WRAPPER_CALL)

        if base_t_cam_and_points is None:
            return None
        
        
        base_t_cam_pnp, image_points_cam1, image_points_cam2, world_obj_points, inliers = base_t_cam_and_points
        line_indices = match_2d_line_segments(
            lines_img1=self.lines_4_images_cam1[idx],
            lines_img2=lines_img2,
            points_img1=image_points_cam1,
            points_img2=image_points_cam2,
            return_indices=True
        )
        #print(f"image points cam1: {image_points_cam1.shape}, min: {np.min(image_points_cam1, axis=0)}")
        #print(f"image points cam2: {image_points_cam2.shape}, min: {np.min(image_points_cam2, axis=0)}")

        matched_lines_img1_2d = np.asarray([self.lines_4_images_cam1[idx][i1] for i1, _ in line_indices])
        matched_lines_img1_3d = np.asarray([self.lines3d[idx][i1] for i1, _ in line_indices])
        matched_lines_img2_2d = np.asarray([lines_img2[i2] for _, i2 in line_indices])

        if self.debug_visualize_matching:
            draw_matched_lines(
                bgr_img1=cv2.cvtColor(self.cam1_bgr_images[idx], cv2.COLOR_BGR2RGB),
                lines_img1=matched_lines_img1_2d,
                bgr_img2=cam2_rgb_image,
                lines_img2=matched_lines_img2_2d,
                #lines_img1_unmatched=self.lines_4_images_cam1[idx],
                #lines_img2_unmatched = lines_img2
            )
            plt.show()


        time_tracker.add_time_stamp(TimeLabels.LINE_MATCHING)

        if matched_lines_img2_2d.shape[0] < 1:
            return base_t_cam_pnp

        if self.debug_visualize_3d:
            visualize_lines_3d(points=self.cam1_xyz_images[idx].reshape(-1, 3), lines=matched_lines_img1_3d)

        cam2_t_base_bundle_adjustment = optimize_pnpl(
            initial_cam_t_base=np.linalg.inv(base_t_cam_pnp).copy(),
            points_3d=world_obj_points[inliers].copy(),
            points_2d=image_points_cam2[inliers].copy(),
            intrinsic_cam_mat=self.cam2_intrinsic_mtx.copy(),
            lines_2d = matched_lines_img2_2d,
            lines_3d = matched_lines_img1_3d,
            config=self.pnpl_optimisation_conf,
            visualize_result= cam2_rgb_image if self.debug_visualize_pnpl else None
        )

        time_tracker.add_time_stamp(TimeLabels.PNL_OPTIMIZATION)

        if fd is not None and base_t_cam_and_points is not None:
            _ , best_image_points_cam1, best_image_points_cam2, points1_3d, inlier_indices = base_t_cam_and_points
            fd.visualize_localizer(
                robot_img_rgb=cv2.cvtColor(self._extract_and_match_wrapper.cam1_bgr_images_for_vis[idx], code=cv2.COLOR_BGR2RGB),
                headset_img_rgb=cam2_rgb_image,
                points1=best_image_points_cam1[inlier_indices],
                points2=best_image_points_cam2[inlier_indices],
                points1_3d=points1_3d[inlier_indices],
                base_t_cam=np.linalg.inv(cam2_t_base_bundle_adjustment),
                robot_lines_2d=matched_lines_img1_2d,
                headset_lines_2d=matched_lines_img2_2d,
                robot_lines_3d=matched_lines_img1_3d,
                headset_intrinsic_mat=self._extract_and_match_wrapper.cam2_mtx
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

        augmented_images = []
        augmented_images_features = []
        map_points_to_unaugmented_functions_and_names = []

        for c_aug in self._extract_and_match_wrapper.crop_augmentations:
            for r_aug in self._extract_and_match_wrapper.rotation_augmentations:
                augmented_image, backward_aug2 = c_aug.forward(cam2_rgb_image)
                augmented_image, backward_aug1 = r_aug.forward(augmented_image)

                augmented_images.append(augmented_image)
                augmented_images_features.append(self._extract_and_match_wrapper.extract_and_match.get_features(augmented_image))
                map_points_to_unaugmented_functions_and_names.append(
                    (f"{c_aug} x {r_aug}", lambda points: backward_aug2(backward_aug1(points)))
                )

        while est_base_t_cam is None and number_tries < number_retry:
            idx = self._extract_and_match_wrapper.sheduler.get_best()
            est_base_t_cam = self._est_base_t_cam2_4_idx(
                idx=idx,
                cam2_rgb_image_features = augmented_images_features,
                backward_transformations_names = map_points_to_unaugmented_functions_and_names,
                cam2_augmented_images = np.asarray(augmented_images),
                cam2_rgb_image = cam2_rgb_image,
                lines_img2 = lines_img2,
                time_tracker = time_tracker,
                fd = fd
            )
            self._extract_and_match_wrapper.sheduler.adjust(idx, est_base_t_cam is not None)
            number_tries += 1
        return est_base_t_cam
    
    @property
    def extract_and_match_wrapper(self)->ExtractAndMatchWrapper | None:
        return self._extract_and_match_wrapper