import cv2
import numpy as np
from predictor_handling import *
from extractors_and_matchers import *
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from pose_estimation.pnpl_optimizer import *
from line_utilities import * 
from dataclasses import dataclass
from numbers import Number

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

            extract_and_match_wrapper_config:ExtractAndMatchWrapperConfig,
            lsd_cleanup_passes_configs:MultiPassLineMergingConfig = MultiPassLineMergingConfig(),
            line_matching_config:LineMatchingConfig = LineMatchingConfig(),
            line_fitting_3d_config:LineFitting3dConfig = LineFitting3dConfig(),
            pnpl_optimisation_conf:PnPLOptimizerConfig = PnPLOptimizerConfig(),
            cam2_lsd_size:None | tuple[int, int] = None,

            debug_visualize_2d:bool = False,
            debug_visualize_pnpl:bool = False,
            debug_visualize_3d:bool = False
        ):
        super().__init__()
        assert assert_intrinsic_mat(cam2_intrinsic_mtx)
        self.cam2_intrinsic_mtx = cam2_intrinsic_mtx

        assert assert_mxnx3_np_uint8_image_batch(cam1_bgr_images)
        assert cam1_xyz_images.shape == cam1_bgr_images.shape
        self.cam1_xyz_images = cam1_xyz_images

        assert isinstance(extract_and_match_wrapper_config, ExtractAndMatchWrapperConfig)
        self.extract_and_match_wrapper = ExtractAndMatchWrapper(
            cam2_mtx=cam2_intrinsic_mtx,
            cam1_bgr_images=cam1_bgr_images,
            cam1_xyz_images=cam1_xyz_images,
            config=extract_and_match_wrapper_config
        )

        assert isinstance(lsd_cleanup_passes_configs, MultiPassLineMergingConfig)
        self.lsd_cleanup_passes = lsd_cleanup_passes_configs.passes

        assert isinstance(line_matching_config, LineMatchingConfig)
        self.line_matching_config = line_matching_config

        self.line_fitting_3d_method = (lambda points3d:(
            line_segment_regression_3d_ransaac(
                xyz_points=points3d,
                inlier_distance=line_fitting_3d_config.ransac_inlier_distance,
                itterations=line_fitting_3d_config.ransac_itterations
            )
        )) if line_fitting_3d_config.use_ransac else lambda points3d: robust_pca_2d_3d_points_lineseg_regression(points=points3d)


        assert isinstance(pnpl_optimisation_conf, PnPLOptimizerConfig)
        self.pnpl_optimisation_conf = pnpl_optimisation_conf


        self.debug_visualize_2d = debug_visualize_2d
        self.debug_visualize_3d = debug_visualize_3d
        self.debug_visualize_pnpl = debug_visualize_pnpl

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.line_seg_detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_NONE)

        self.cam1_bgr_images = cam1_bgr_images

        self.lines_4_images_cam1 = [
            self.lsd_and_cleanup_on_image(img) for img in cam1_bgr_images
        ]

        self.cam2_lsd_size = cam2_lsd_size

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
        """
        :param img1: HxWx3 BGR image as numpy array
        :param img2: HxWx3 RGB image as numpy array
        :param lines1: Nx4 array of line segments
        :param lines2: Nx4 array of line segments
        :param line_pairs: Nx2x4 array of matched line pairs
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
    
    def visualize_features_3d(
            self,
            point_cloud:np.ndarray,
            line_points_3d:np.ndarray,
            line_segments_3d:np.ndarray
        ):
        """
        Visualizes the inputs in 3d in relation to a base frame
        :param point_cloud: Nx3 numpy array of xyz-points
        :param line_points_3d: LxMx3 numpy array of xyz-points
        :param line_segments_3d: Lx6 numpy array of lines with each line: [x1, y1, z1, x2, y2, z2]
        """
        assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 3
        assert line_points_3d.ndim == 3 and line_points_3d.shape[-1] == 3
        assert line_segments_3d.ndim == 2 and line_segments_3d.shape[-1] == 6

        import open3d as o3d

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        pcd.paint_uniform_color([0, 0, 0])

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        
        to_vis = [pcd, base_frame]

        if len(line_points_3d) > 0:
            pcd_lines = o3d.geometry.PointCloud()
            pcd_lines.points = o3d.utility.Vector3dVector(np.concatenate(line_points_3d, axis=0))
            pcd_lines.paint_uniform_color([1, 0, 0])
            to_vis.append(pcd_lines)

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(line_segments_3d.reshape(-1,3))
        line_set.lines = o3d.utility.Vector2iVector(np.array([[i, i+1] for i in range(line_segments_3d.shape[0]*2-1) if i % 2 == 0]))
        to_vis.append(line_set)

        o3d.visualization.draw_geometries(to_vis, f"3D features visualization")


    def est_base_t_cam2_4_idx(
            self,
            cam2_rgb_image: np.ndarray,
            idx:int,
            lines_img2:np.ndarray,
            time_tracker:TimeTracker = TimeTracker()
        ) -> np.ndarray | None:

        time_tracker.reset_elapsed_time()
        base_t_cam_and_points = self.extract_and_match_wrapper.est_base_t_cam2_and_points(
            idx=idx, cam2_rgb_image=cam2_rgb_image
        )
        time_tracker.add_time_stamp("estimate base_t_cam2 with points")

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

        time_tracker.add_time_stamp("Line 2d to 3d transformation")

        if self.debug_visualize_2d:
            self.visualize_features_2d(
                img1=self.cam1_bgr_images[idx],
                img2=cam2_rgb_image,
                lines1_processed=lines_img1,
                lines2_processed=lines_img2,
                points1=image_points_cam1,
                points2=image_points_cam2
            )
        if self.debug_visualize_3d:
            self.visualize_features_3d(
                point_cloud=self.cam1_xyz_images[idx].reshape(-1,3), 
                line_points_3d=[line_seg_2d_to_3d_points(line_pair[0], self.cam1_xyz_images[idx]) for line_pair in line_pairs],
                line_segments_3d=matched_lines_3d
            )
        
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

        time_tracker.add_time_stamp("Bundle adjustment")
        time_tracker.print_report()

        return np.linalg.inv(cam2_t_base_bundle_adjustment)



    def cleanup_lines(self, lines):
        lines = lines
        for ref_conf in self.lsd_cleanup_passes:
            lines = remove_short_2d_line_segments(
                merge_close_line_segments(
                    lines,ref_conf.max_angle_diff_deg,ref_conf.max_midpoint_dist_px, ref_conf.max_endpoint_dist_px, ref_conf.use_quick_merge
                ),
            min_line_length_px= ref_conf.min_line_length)
        return lines
    

    def lsd_and_cleanup_on_image(self, bgr_image:np.ndarray, lsd_at_size:None | tuple[int, int] = None)->np.ndarray:
        """
        Runs LSD on an image that can be scaled down beforehand, then cleans those lines up and scales them back
        :param bgr_image: The HxWx3-uint8 BGR image to be done lsd upon
        :param lsd_at_size: None or a tuple: (height, width) in px
        :return Nx4 line array of the format: [[x1, y1, x2, y2], ...]
        """
        assert assert_mxnx3_np_uint8_image(bgr_image)

        cam2_grey_img = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        if lsd_at_size is None:
            return self.cleanup_lines(self.line_seg_detector.detect(cam2_grey_img)[0].squeeze(1))
        
        # In case of scaling:
        h_orig, w_orig = bgr_image.shape[:2]
        h_scaled, w_scaled = lsd_at_size

        cam2_image_scaled = cv2.resize(cam2_grey_img, (w_scaled, h_scaled), interpolation = cv2.INTER_AREA)
        lines_scaled_raw = self.line_seg_detector.detect(cam2_image_scaled)[0].squeeze(1)
        lines_scaled = self.cleanup_lines(lines_scaled_raw)

        if lines_scaled.size > 0:
            lines_scaled[:, [0,2]] *= w_orig/w_scaled
            lines_scaled[:, [1,3]] *= h_orig/h_scaled
        return lines_scaled

    

    def est_base_t_cam2(self,cam2_bgr_image: np.ndarray, number_retry:int = 2, time_tracker:TimeTracker = TimeTracker()) -> np.ndarray | None:
        number_tries = 0
        est_base_t_cam = None
        cam2_rgb_image = cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)


        time_tracker.reset_elapsed_time()
        lines_img2 = self.lsd_and_cleanup_on_image(cam2_bgr_image, lsd_at_size=self.cam2_lsd_size)
        time_tracker.add_time_stamp("Image 2 LSD + cleanup")

        while est_base_t_cam is None and number_tries < number_retry:
            idx = self.extract_and_match_wrapper.sheduler.get_best()
            est_base_t_cam = self.est_base_t_cam2_4_idx(
                idx=idx,
                lines_img2 = lines_img2,
                cam2_rgb_image = cam2_rgb_image,
            )
            self.extract_and_match_wrapper.sheduler.adjust(idx, est_base_t_cam is not None)
            number_tries += 1
        return est_base_t_cam
    
    def update_pose(self,cam2_bgr_image: np.ndarray, rough_base_t_cam2:np.ndarray, time_tracker:TimeTracker) -> np.ndarray | None:
        return self.est_base_t_cam2(cam2_bgr_image=cam2_bgr_image, number_retry=1, time_tracker=time_tracker)





if __name__ == "__main__":
    robot_data = RobotEnvironment.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data_r")
    headset_data = HeadsetData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data_h")
    
    
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
        debug_visualize_2d=False, 
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
    