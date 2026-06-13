import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from dataclasses import dataclass

from shared.assertion_helpers import *

from geometric_utilities.union_find import UnionFind
from geometric_utilities.pypose_pne_optimizer import *
from geometric_utilities.ellipsoid_utilities_numpy import *
from geometric_utilities.pne_delta_pose_otimizer import *

from geometric_utilities.foreground_segmentation import Segmenter, display_image_masks, YOLOv26Segmenter, Sam3Prompt, SAM3Segmenter
from small_utilities.image_augmentation import *
from small_utilities.sheduler import *
from predictor_handling import *
from extractors_and_matchers import *
from geometric_utilities.packed_bool_mask_storage import ImageMaskStorage



@dataclass(kw_only=True, frozen=True)
class PointCloudMatchingConfig:
    max_centroid_dist:float | None = 0.05
    max_color_dist:float | None = 500
    min_cluster_size:int = 1
    min_number_points_per_detected_object:int = 1000



def match_point_clouds(
        masks:ImageMaskStorage, 
        xyz_images:np.ndarray, 
        mask_image_indices:list[int],
        point_cloud_matching_config:PointCloudMatchingConfig,
        bgr_images:np.ndarray | None = None
    )->list[list[int]]:

    n = 0
    object_avg_colors = []
    object_avg_centers = []
    index_map = []

    for pc_idx, imgidx in enumerate(mask_image_indices):
        mask = masks.see_mask(pc_idx)

        pc_3d = xyz_images[imgidx][mask > 0]
        pc_3d = pc_3d[np.isfinite(pc_3d).all(axis=-1)]

        if pc_3d.shape[0] < point_cloud_matching_config.min_number_points_per_detected_object:
            continue

        n += 1
        index_map.append(pc_idx)

        if point_cloud_matching_config.max_centroid_dist is not None:
            object_avg_centers.append(np.mean(pc_3d, axis = 0))

        if point_cloud_matching_config.max_color_dist is not None:
            pc_color = bgr_images[imgidx][mask > 0]
            object_avg_colors.append(np.mean(pc_color, axis = 0))

    valid_mask = np.ones((n, n), dtype=bool)

    if point_cloud_matching_config.max_centroid_dist is not None:
        object_avg_centers = np.stack(object_avg_centers, axis = 0)

        centroid_distance_matrix = np.linalg.norm(
            object_avg_centers[:, None] - object_avg_centers[None, :], axis=-1
        )
        valid_mask &= (centroid_distance_matrix < point_cloud_matching_config.max_centroid_dist)

    if point_cloud_matching_config.max_color_dist is not None:
        object_avg_colors = np.stack(object_avg_colors, axis = 0)

        color_distance_matrix = np.mean(
            np.abs(
                object_avg_colors[:, None] -
                object_avg_colors[None, :]
            ),
            axis=-1
        )
        valid_mask &= (color_distance_matrix < point_cloud_matching_config.max_color_dist)

    union_find = UnionFind(n)
    for row_idx in range(n):
        for col_idx in range(row_idx+1, n):
            if valid_mask[row_idx, col_idx]:
                union_find.union(row_idx, col_idx)

    clusters = union_find.return_clusters()
    clusters = [c for c in clusters if len(c) >= point_cloud_matching_config.min_cluster_size]

    clusters_old_indices = [[index_map[i] for i in c] for c in clusters]
    return clusters_old_indices



def images_to_primal_quadratics(
        bgr_images:np.ndarray,
        xyz_images:np.ndarray,
        segmenter:Segmenter,
        matching_config:PointCloudMatchingConfig = PointCloudMatchingConfig(),
        fitting_config:EllipsoidFittingConfig = EllipsoidFittingConfig(),
        debug_vis_masks:bool = False,
)-> tuple[np.ndarray, np.ndarray]:
    """
    Generates M ellipsoids from an environment
    :param bgr_images: An NxHxWx3-uint8 array of BGR images
    :param xyz_images: An NxHxWx3-float array of 3d points in the base frame
    :param segmenter: The object to detect objects in an image
    :param iforest_contamination: The contamination before fitting each shape
    :param max_centroid_dist: The maximum distance in meters between two object centers in different images to be considered the same
    :return: the base_t_ellipsoid hom. matrices (Mx4x4) and the primal quadratics (Mx4x4) and the average colors (Nx3) (0-255)
    """
    assert all([assert_mxnx3_np_uint8_image(bgr_img) for bgr_img in bgr_images])
    assert xyz_images.ndim == 4 and xyz_images.shape[-1] == 3
    assert bgr_images.shape[:3] == xyz_images.shape[:3]

    # Generate objects

    # (n-all-objs)xHxW-bool masks
    mask_storage = ImageMaskStorage(bgr_images.shape[1], bgr_images.shape[2])


    object_image_idxs = []
    for i, (bgr_img, xyz_img) in enumerate(zip(bgr_images, xyz_images)):
        object_masks = segmenter.get_object_masks(bgr_image=bgr_img, visualize=debug_vis_masks)
        for object_mask in object_masks:
            mask_storage.add_new_mask(object_mask)
            object_image_idxs.append(i)
    
    if len(object_image_idxs) < 1:
        return np.empty((0,4,4)), np.empty((0,4,4))


    clusters = match_point_clouds(
        masks=mask_storage,
        xyz_images=xyz_images,
        mask_image_indices=object_image_idxs,
        point_cloud_matching_config=matching_config,
        bgr_images=bgr_images
    )

    fused_base_t_ellipsoid_s = []
    fused_primal_quaddratic_s = []
    for i,cluster in enumerate(clusters):
        b_t_e__p_q = fit_ellipsoid_to_3d_point_cloud(
            np.concatenate(
                [xyz_images[object_image_idxs[i]][mask_storage.see_mask(i)] for i in cluster],
                axis = 0
            ),
            config=fitting_config
        )
        if b_t_e__p_q is None:
            continue
        b_t_e, p_q = b_t_e__p_q
        fused_base_t_ellipsoid_s.append(b_t_e)
        fused_primal_quaddratic_s.append(p_q)
    fused_base_t_ellipsoid_s = np.array(fused_base_t_ellipsoid_s)
    fused_primal_quaddratic_s = np.array(fused_primal_quaddratic_s)

    return fused_base_t_ellipsoid_s, fused_primal_quaddratic_s


def image_to_primal_conics(
        bgr_image:np.ndarray, 
        segmenter:Segmenter,
        debug_vis_masks:bool = True
    ) -> np.ndarray:
    """
    Creates a Nx3x3 batch of primal conics from the image by segmenting it using sam3.
    :param bgr_image: the bgr image
    :param segmenter: how to segment the image
    :param debug_vis_masks: If true will show the used mask
    :return: the primal conics (Nx3x3) and avg colors or None
    """
    #tt = TimeTracker()
    assert assert_mxnx3_np_uint8_image(bgr_image)

    object_masks = segmenter.get_object_masks(bgr_image, visualize=debug_vis_masks)

    if object_masks is None or object_masks.shape[0] == 0:
        return np.empty((0,3,3))

    primal_conic_s = []

    for object_mask in object_masks:
        rows, cols = np.where(object_mask)
        pc_2d = np.column_stack((cols, rows))
        primal_conic_s.append(fit_primal_conic_to_2d_point_cloud(pc_2d))

    if debug_vis_masks:
        mu_s, sigma_s = primal_conics_to_gaussian_ellipses(np.array(primal_conic_s))
        e_s = gaussian_ellipse_s_to_matplotlib_ellipse_s(gauss_ellipse_batch_tuple_to_mat_batch(mu_s=mu_s, sigma_s=sigma_s), line_style="-")
        fig, ax = plt.subplots(figsize = (12,8))
        for e in e_s:
            ax.add_patch(e)
        ax.imshow(bgr_image)
        plt.show()

    return np.array(primal_conic_s)    


def visualize_pose_prediction(
        fd:FeatureDrawing,
        dual_quadratics:np.ndarray,
        cam_t_base:np.ndarray,
        intrinsic_mtx:np.ndarray,
        obs_gaussians_sigma_mu_s:np.ndarray,
        proj_match_indices:list[int],
        obs_match_indices:list[int]
    ):
    """
    :param fd: The feature drawing with the axis to draw upon and the style guide
    :param dual_quadratics: The dual quadratics to project (Nx4x4)
    :param cam_t_base: The 4x4 SE3 cam-T_base matrix
    :param intrinsic_mtx: The 3x3 intrinsic matrix
    :param obs_gaussians_sigma_s: Mx2x3 array of gaussians: [[sigma_0 | mu_0], ...]
    :param proj_match_indices: List of length n, where dual_quadratics[proj_match_indices[i]] ~ obs_gaussians_sigma_mu_s[obs_match_indices[i]]
    :param obs_match_indices: List of length n
    """
    
    proj_primal_conincals = project_primal_quadratics_to_primal_conicals(
        primal_quadratics=np.linalg.inv(dual_quadratics),
        cam_t_base=cam_t_base,
        intrinsic_mtx=intrinsic_mtx
    )
    proj_mu_s, proj_sigma_s = primal_conics_to_gaussian_ellipses(proj_primal_conincals)
    proj_sigma_mu_s = gauss_ellipse_batch_tuple_to_mat_batch(mu_s=proj_mu_s, sigma_s=proj_sigma_s)


    n_obs = obs_gaussians_sigma_mu_s.shape[0]
    n_proj = proj_sigma_mu_s.shape[0]

    unmatched_obs = list(set(range(n_obs))-set(obs_match_indices))
    unmatched_proj = list(set(range(n_proj))-set(proj_match_indices))
    # Plot unmatched ones:
    ellipses_unmatched = gaussian_ellipse_s_to_matplotlib_ellipse_s(
        gaussian_ellipse_s=proj_sigma_mu_s[unmatched_proj],
        colors=fd.sc.unmatched_color,
        line_style=fd.sc.proj_line_style
    ) + gaussian_ellipse_s_to_matplotlib_ellipse_s(
        gaussian_ellipse_s=obs_gaussians_sigma_mu_s[unmatched_obs],
        colors=fd.sc.unmatched_color,
        line_style=fd.sc.obs_line_style
    )
    for e in ellipses_unmatched:
        fd.ax.add_patch(e)


    # Plot matched ones
    n = len(obs_match_indices)
    colors = plt.cm.jet(np.linspace(0,1, n))
    proj_ellipses_matched = gaussian_ellipse_s_to_matplotlib_ellipse_s(
        gaussian_ellipse_s=proj_sigma_mu_s[proj_match_indices],
        colors=colors,
        line_style=fd.sc.proj_line_style
    )
    obs_ellipses_matched = gaussian_ellipse_s_to_matplotlib_ellipse_s(
        gaussian_ellipse_s=obs_gaussians_sigma_mu_s[obs_match_indices],
        colors=colors,
        line_style=fd.sc.obs_line_style
    )
    for proj_e, obs_e in zip(proj_ellipses_matched, obs_ellipses_matched):
        fd.ax.add_patch(proj_e)
        fd.ax.add_patch(obs_e)
    fd.ax.scatter(
        obs_gaussians_sigma_mu_s[obs_match_indices,0,2],
        obs_gaussians_sigma_mu_s[obs_match_indices,1,2],
        color=colors, s=fd.sc.point_size, alpha=fd.sc.point_alpha, marker = fd.sc.obs_point_style)
    
    fd.ax.scatter(
        proj_sigma_mu_s[proj_match_indices,0,2],
        proj_sigma_mu_s[proj_match_indices,1,2],
        color=colors, s=fd.sc.point_size, alpha=fd.sc.point_alpha, marker = fd.sc.proj_point_style)
    
    fd.ax.quiver(
        proj_sigma_mu_s[proj_match_indices,0,2],
        proj_sigma_mu_s[proj_match_indices,1,2],
        obs_gaussians_sigma_mu_s[obs_match_indices,0,2]-proj_sigma_mu_s[proj_match_indices,0,2], 
        obs_gaussians_sigma_mu_s[obs_match_indices,1,2]-proj_sigma_mu_s[proj_match_indices,1,2],
        angles='xy', scale_units='xy', scale=1,
        color=colors,
        alpha=fd.sc.arrow_alpha,
        width=0.005
    )
    
class EllipsoidPredictor(PosePredictor):
    def __init__(
            self,
            cam2_intrinsic_mtx:np.ndarray,
            cam1_bgr_images:np.ndarray,
            cam1_xyz_images:np.ndarray,
            time_tracker_init:TimeTracker | None = None,
            extract_and_match_wrapper_config:ExtractAndMatchWrapperConfig = ExtractAndMatchWrapperConfig(),
            cam1_segmenter:Segmenter | None = None,
            cam2_segmenter:Segmenter | None = None,
            pne_optimizer:PnEOptimizer | None = None,
            min_number_matched_ellipsoids_for_opt:int = 1,
            ellipsoid_refinement_at_res: None | tuple[int, int] = None,
            matching_config:GaussianMatchingConfig = GaussianMatchingConfig(),
            ellipsoid_matching_config:PointCloudMatchingConfig = PointCloudMatchingConfig(),
            ellipsoid_fitting_config:EllipsoidFittingConfig = EllipsoidFittingConfig(),
            visualize_pne_optimisation:bool = False,
            visualize_matching:bool = False,
            visualize_environment_generation:bool = False,
            visualize_segmentation_masks:bool = False,
        ):
        """
        Creates an Predictor that uses ellipsoids as features
        :param cam2_intrinsic_mtx: The 3x3 intrinsic matrix for camera 2
        :param cam1_bgr_images: BxHxWx3-uint8 array of bgr images for camera 1
        :param cam1_xyz_images: BxHxWx3-float array of xyz-point images for camera 1 in the base ref. frame
        :param time_tracker_init: A timetracker where important steps during the initialization will be registered
        :param extract_and_match_wrapper_config: The config for the initial point based prediction
        :param cam1_segmenter: The tool to detect objects in the images during initialisation (Standard Sam3Segmenter if left to None)
        :param cam2_segmenter: The tool to detect objects in the images after initialisation (Standard YOLOv26Segmenter if left to None)
        :param pne_optimizer: The optimizer to optimize a pose given ellipsoids
        :param min_number_matched_ellipsoids_for_opt: How many ellipsoids have to be matched between cam1 & cam2 image to do optimisation
        :param ellipsoid_refinement_at_res: If not None cam2 images will be scaled to this resolution for anything ellipsoid related (can boost runtime)
        :param matching_config: How to match the observed and projected gaussians
        :param ellipsoid_matching_config: How to match the point clouds from different perspectives to get one 
        :param ellipsoid_fitting_config: How to fit ellipsoids to pointclouds
        :param visualize_pne_optimisation: If True will visualize the pne optimizer calls
        :param visualize_matching: If True will visualize the obs & proj ellipsoid matching
        :param visualize_ellipsoid_fitting: If True will visualize the fitting of each 3d ellipsoid
        :param visualize_environment_generation: If True will visualize the environment -> 3d ellipsoid generation
        :param visualize_segmentation_masks: If True will visualise the masks generated by the Segmenters
        """
        super().__init__()
        assert assert_intrinsic_mat(cam2_intrinsic_mtx)
        
        self.cam2_segmenter = cam2_segmenter
        self.pne_optimizer = pne_optimizer

        if cam1_segmenter is None:
            cam1_segmenter = SAM3Segmenter(Sam3Prompt())

        if self.cam2_segmenter is None:
            self.cam2_segmenter = YOLOv26Segmenter()
        if self.pne_optimizer is None:
            self.pne_optimizer = PyposePNEOptimizer()
        if time_tracker_init is None:
            time_tracker_init = TimeTracker()

        self.cam2_intrinsic_mtx = cam2_intrinsic_mtx

        self.ellipsoid_refinement_res = ellipsoid_refinement_at_res

        assert min_number_matched_ellipsoids_for_opt > 0
        self.min_number_matched_ellipsoids_for_opt = min_number_matched_ellipsoids_for_opt

        self.visualize_segmentation_masks = visualize_segmentation_masks
        time_tracker_init.reset_elapsed_time()
        b_t_e_s, prim_quad_s = images_to_primal_quadratics(
            bgr_images=cam1_bgr_images,
            xyz_images=cam1_xyz_images,
            segmenter=cam1_segmenter,
            matching_config=ellipsoid_matching_config,
            fitting_config=ellipsoid_fitting_config,
            debug_vis_masks=self.visualize_segmentation_masks,
        )
        self.base_t_ellipsoid_s = b_t_e_s
        self.primal_quadratic_s = prim_quad_s

        if visualize_environment_generation:
            visualize_primal_quadratics(
                base_t_ellipsoid_s=b_t_e_s,
                primal_quadratic_s=prim_quad_s,
                bg_point_cloud=cam1_xyz_images.reshape(-1,3),
                bg_point_cloud_colors=cam1_bgr_images.reshape(-1,3),
            )

        time_tracker_init.add_time_stamp("Primal quadratics creation")



        self.extract_and_match_wrapper = ExtractAndMatchWrapper(
            cam2_mtx=cam2_intrinsic_mtx,
            cam1_bgr_images=cam1_bgr_images,
            cam1_xyz_images=cam1_xyz_images,
            config=extract_and_match_wrapper_config
        )
        time_tracker_init.add_time_stamp("Extract and match wrapper initialisation")

        self.cam1_image_size = cam1_bgr_images.shape[1:3]
        self.matching_config = matching_config

        self.visualize_pne_optimisation = visualize_pne_optimisation
        self.visualize_matching = visualize_matching
        

    @staticmethod
    def get_creation_function(
            cam2_intrinsic_mtx:np.ndarray,
            extract_and_match_wrapper_config:ExtractAndMatchWrapperConfig = ExtractAndMatchWrapperConfig(),
            cam1_segmenter:Segmenter | None = None,
            cam2_segmenter:Segmenter | None = None,
            pne_optimizer:PnEOptimizer | None = None,
            min_number_matched_ellipsoids_for_opt:int = 1,
            ellipsoid_refinement_at_res: None | tuple[int, int] = None,
            matching_config:GaussianMatchingConfig = GaussianMatchingConfig(),
            ellipsoid_matching_config:PointCloudMatchingConfig = PointCloudMatchingConfig(),
            ellipsoid_fitting_config:EllipsoidFittingConfig = EllipsoidFittingConfig(),
            visualize_pne_optimisation:bool = False,
            visualize_matching:bool = False,
            visualize_environment_generation:bool = False,
            visualize_segmentation_masks:bool = False
    ):
        """
        Returns a function with which a new EllipsoidPredictor may be created.
        For parameter info look at `__init__`
        :return: f(robot_env,time_tracker) -> EllipsoidPredictor
        """
        creation_function = lambda robot_env, init_tt: EllipsoidPredictor(
            cam2_intrinsic_mtx = cam2_intrinsic_mtx,
            cam1_bgr_images = robot_env.robot_bgr_images,
            cam1_xyz_images = robot_env.robot_xyz_images,
            time_tracker_init = init_tt,
            extract_and_match_wrapper_config = extract_and_match_wrapper_config,
            cam1_segmenter = cam1_segmenter,
            cam2_segmenter = cam2_segmenter,
            pne_optimizer = pne_optimizer,
            min_number_matched_ellipsoids_for_opt = min_number_matched_ellipsoids_for_opt,
            ellipsoid_refinement_at_res = ellipsoid_refinement_at_res,
            matching_config = matching_config,
            ellipsoid_matching_config = ellipsoid_matching_config,
            ellipsoid_fitting_config = ellipsoid_fitting_config,
            visualize_pne_optimisation = visualize_pne_optimisation,
            visualize_matching = visualize_matching,
            visualize_environment_generation = visualize_environment_generation,
            visualize_segmentation_masks = visualize_segmentation_masks
        )
        return creation_function


    def est_base_t_cam2(self,cam2_bgr_image: np.ndarray, number_retry:int = 1, time_tracker:TimeTracker = TimeTracker(), fd:FeatureDrawing | None = None) -> np.ndarray | None:
        """
        Estimates the hom. transformation: baseT_cam2 based on point features and then refines it using ellipsoids
        :param cam2_bgr_image: The camera 2 image (HxWx3-uint8 array)
        :param number_retry: With how many different cam1 images the prediction may be tried (upper bound)
        :param time_tracker: A time tracker where timestamps for the different subcomponents will be added.
        :return: The 4x4 Pose in SE3 if prediction was successful else None
        """
        time_tracker.reset_elapsed_time()
        est_base_t_cam = self.extract_and_match_wrapper.est_base_t_cam2_with_retry(
            cam2_bgr_image=cam2_bgr_image, 
            number_retry=number_retry,
            fd = fd
        )
        time_tracker.add_time_stamp("Point based initial guess")
        if est_base_t_cam is None:
            return None

        optimized_cam_t_base = self.update_pose(
            cam2_bgr_image=cam2_bgr_image,
            rough_cam_t_base=np.linalg.inv(est_base_t_cam),
            time_tracker=time_tracker,
            fd=fd
        )
        if optimized_cam_t_base is None:
            return est_base_t_cam
        
        return np.linalg.inv(optimized_cam_t_base)
    

    def update_pose(self,cam2_bgr_image: np.ndarray, rough_cam_t_base:np.ndarray, time_tracker:TimeTracker, fd:FeatureDrawing | None = None) -> np.ndarray | None:
        """
        Optimizes a given pose using ellipsoids
        :param cam2_bgr_image: HxWx3 bgr image
        :param rough_base_t_cam2: A rough base_t_cam2 estimate.
        :param time_tracker: a time-tracker object, that will be used by the Pose Predictor to note the runtimes
        """
        if self.base_t_ellipsoid_s.shape[0] < self.min_number_matched_ellipsoids_for_opt:
            return None
        
        scaled_cam2_intrinsics = self.cam2_intrinsic_mtx.copy()
        if self.ellipsoid_refinement_res is not None:
            h_old, w_old = cam2_bgr_image.shape[:2]
            w_new, h_new = self.ellipsoid_refinement_res
            cam2_bgr_image = cv2.resize(cam2_bgr_image, (w_new, h_new))
            scaled_cam2_intrinsics[0, :] *= w_new/w_old
            scaled_cam2_intrinsics[1, :] *= h_new/h_old

        time_tracker.reset_elapsed_time()

        proj_primal_conics = project_primal_quadratics_to_primal_conicals(
            primal_quadratics= self.primal_quadratic_s,
            cam_t_base=rough_cam_t_base,
            intrinsic_mtx=scaled_cam2_intrinsics,
        )

        proj_primal_conics = filter_good_primal_conicals(
            primal_conicals=proj_primal_conics
        )

        proj_gauss_elli_mu, proj_gauss_elli_sigmas = primal_conics_to_gaussian_ellipses(proj_primal_conics)
        proj_gaussian_ellipses = gauss_ellipse_batch_tuple_to_mat_batch(proj_gauss_elli_mu, proj_gauss_elli_sigmas)

        time_tracker.add_time_stamp("Projecting the 3d ellipsoids to 2d Gauss")

        observed_primal_conics = image_to_primal_conics(
            bgr_image=cam2_bgr_image, 
            segmenter=self.cam2_segmenter,
            debug_vis_masks=self.visualize_segmentation_masks,
        )
        time_tracker.add_time_stamp("Image to primal conics")

        if observed_primal_conics.shape[0] < self.min_number_matched_ellipsoids_for_opt:
            return None

        obs_gauss_elli_mu, obs_gauss_elli_sigmas = primal_conics_to_gaussian_ellipses(observed_primal_conics)

        obs_gauss_ellipses = gauss_ellipse_batch_tuple_to_mat_batch(obs_gauss_elli_mu, obs_gauss_elli_sigmas)
        time_tracker.add_time_stamp("Primal conics to gaussians")

        proj_match_idx_s, obs_match_idx_s = match_gaussians_hungarian_on_wasserstein(
            proj_sigma_mu_s=proj_gaussian_ellipses,
            obs_sigma_mu_s=obs_gauss_ellipses,
            image_size=cam2_bgr_image.shape[:2],
            config=self.matching_config,
            visualize_matching=cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB) if self.visualize_matching else None
        )
        time_tracker.add_time_stamp("Matching the 2d gaussians")

        if proj_match_idx_s.shape[0] < self.min_number_matched_ellipsoids_for_opt:
            print(f"to few ellipsoids for optimisation: {proj_match_idx_s.shape[0]}")
            return None

        cam2_t_base_opt = self.pne_optimizer.optimize_pne(
            initial_cam_t_base=rough_cam_t_base,
            primal_quadratics=self.primal_quadratic_s[proj_match_idx_s],
            primal_conicals= np.array(observed_primal_conics)[obs_match_idx_s],
            intrinsic_cam_mat=scaled_cam2_intrinsics,
            visualize_result= cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB) if self.visualize_pne_optimisation else None,
        )
        
        time_tracker.add_time_stamp("PNE optimisation")

        if fd is not None:
            visualize_pose_prediction(
                fd=fd,
                dual_quadratics=np.linalg.inv(self.primal_quadratic_s),
                cam_t_base=cam2_t_base_opt if cam2_t_base_opt is not None else rough_cam_t_base,
                intrinsic_mtx=self.cam2_intrinsic_mtx,
                obs_gaussians_sigma_mu_s=obs_gauss_ellipses,
                proj_match_indices=list(proj_match_idx_s),
                obs_match_indices=list(obs_match_idx_s)
            )

        return cam2_t_base_opt