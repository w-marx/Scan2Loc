import cv2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

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


class ImageMaskStorage:
    """
    Stores boolean masks memory efficient (1 bit per bool)
    """

    def __init__(self, height:int, width:int):
        self.h = height
        self.w = width
        self.storage = []
        self.id_storage = {}
        self.c_idx = 0

    def add_new_mask(self,mask:np.ndarray):
        """
        Adds the uint8/bool mask to storage
        """
        assert mask.shape == (self.h, self.w), f"got {mask.shape} instead of {self.h},{self.w}"

        list_idx = int(self.c_idx/8)
        bit_idx = self.c_idx % 8

        if bit_idx == 0:
            self.storage.append(np.zeros((self.h, self.w), dtype = np.uint8))
        if mask.dtype != bool:
            mask = (mask != 0).astype(np.uint8)

        self.storage[list_idx] |= mask << bit_idx

        self.c_idx += 1
    
    def see_mask(self, idx:int)->np.ndarray:
        assert idx < self.c_idx
        list_idx = int(idx/8)
        bit_idx = idx % 8
        return ((self.storage[list_idx] >> bit_idx) & 1).astype(bool)






def images_to_primal_quadratics(
        bgr_images:np.ndarray,
        xyz_images:np.ndarray,
        segmenter:Segmenter,
        max_centroid_dist: float = 0.05,
        max_color_dist:float = 500,
        min_number_points_per_detected_object:int = 1000,
        min_cluster_size:int = 1,
        debug_vis_3d:bool = False,
        debug_vis_masks:bool = False
)-> tuple[np.ndarray, np.ndarray]:
    """
    Generates M ellipsoids from an environment
    :param bgr_images: An NxHxWx3-uint8 array of BGR images
    :param xyz_images: An NxHxWx3-float array of 3d points in the base frame
    :param segmenter: The object to detect objects in an image
    :param max_centroid_dist: The maximum distance in meters between two object centers in different images to be considered the same
    :return: the base_t_ellipsoid hom. matrices (Mx4x4) and the primal quadratics (Mx4x4) and the average colors (Nx3) (0-255)
    """
    assert all([assert_mxnx3_np_uint8_image(bgr_img) for bgr_img in bgr_images])
    assert xyz_images.ndim == 4 and xyz_images.shape[-1] == 3
    assert bgr_images.shape[:3] == xyz_images.shape[:3]

    # Generate objects

    # (n-all-objs)xHxW-bool masks
    mask_storage = ImageMaskStorage(bgr_images.shape[1], bgr_images.shape[2])


    object_avg_colors = []
    object_avg_centers = []
    object_image_idxs = []

    for i, (bgr_img, xyz_img) in enumerate(zip(bgr_images, xyz_images)):
        object_masks = segmenter.get_object_masks(bgr_image=bgr_img)
        for object_mask in object_masks:
            pc = xyz_img[object_mask > 0]
            if pc.shape[0] < min_number_points_per_detected_object:
                continue
            mask_storage.add_new_mask(object_mask)
            object_avg_colors.append(np.mean(bgr_img[object_mask > 0], axis = 0))
            object_avg_centers.append(np.mean(pc, axis = 0))
            object_image_idxs.append(i)
        if debug_vis_masks:
            display_image_masks(bgr_img=bgr_img, masks=object_masks)
    
    object_avg_centers = np.stack(object_avg_centers, axis = 0)
    object_avg_colors = np.stack(object_avg_colors, axis = 0)


    # Join similar objects
    n = object_avg_colors.shape[0]

    # NxN adjecency matrix

    centroid_distance_matrix = np.linalg.norm(
        object_avg_centers[:, None] - object_avg_centers[None, :], axis=-1
    )
    centroid_distance_matrix_mask = centroid_distance_matrix < max_centroid_dist

    color_distance_matrix = np.mean(np.sum(np.abs(
        object_avg_colors[:, None] - object_avg_colors[None, :]
    ), axis = -1), axis = -1)
    color_distance_matrix_mask = color_distance_matrix < max_color_dist
    join_mask = centroid_distance_matrix_mask & color_distance_matrix_mask

    union_find = UnionFind(n)
    for row_idx in range(n):
        for col_idx in range(row_idx+1, n):
            if join_mask[row_idx, col_idx]:
                union_find.union(row_idx, col_idx)

    fused_base_t_ellipsoid_s = []
    fused_primal_quaddratic_s = []
    for i,cluster in enumerate(union_find.return_clusters()):
        if len(cluster) < min_cluster_size:
            continue
        b_t_e, p_q = fit_ellipsoid_to_3d_point_cloud(
            np.concatenate(
                [xyz_images[object_image_idxs[i]][mask_storage.see_mask(i)] for i in cluster],
                axis = 0
            ),
            visualize=False
        )
        fused_base_t_ellipsoid_s.append(b_t_e)
        fused_primal_quaddratic_s.append(p_q)
    fused_base_t_ellipsoid_s = np.array(fused_base_t_ellipsoid_s)
    fused_primal_quaddratic_s = np.array(fused_primal_quaddratic_s)
    
    if debug_vis_3d:
        visualize_primal_quadratics(
            base_t_ellipsoid_s=fused_base_t_ellipsoid_s,
            primal_quadratic_s=fused_primal_quaddratic_s,
            bg_point_cloud=xyz_images.reshape(-1,3),
            bg_point_cloud_colors=bgr_images.reshape(-1,3),
        )

    return fused_base_t_ellipsoid_s, fused_primal_quaddratic_s


def image_to_primal_conics(
        bgr_image:np.ndarray, 
        segmenter:Segmenter, 
    ) -> np.ndarray:
    """
    Creates a Nx3x3 batch of primal conics from the image by segmenting it using sam3.
    :param bgr_image: the bgr image
    :param segmenter: how to segment the image
    :param return_avg_color: If true will compute the avg. color of each conic
    :return: the primal conics (Nx3x3) and avg colors or None
    """
    #tt = TimeTracker()
    assert assert_mxnx3_np_uint8_image(bgr_image)

    object_masks = segmenter.get_object_masks(bgr_image)
    #tt.add_time_stamp("segmenting")

    if object_masks is None or object_masks.shape[0] == 0:
        return np.empty((0,3,3))

    primal_conic_s = []

    for object_mask in object_masks:
        rows, cols = np.where(object_mask)
        pc_2d = np.column_stack((cols, rows))
        primal_conic_s.append(fit_primal_conic_to_2d_point_cloud(pc_2d))
    return np.array(primal_conic_s)    


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
            visualize_pne_optimisation:bool = False,
            visualize_matching:bool = False
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
        :param visualize_pne_optimisation: If True will visualize the pne optimizer calls
        :param visualize_matching: If True will visualize the obs & proj ellipsoid matching
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


        time_tracker_init.reset_elapsed_time()
        b_t_e_s, prim_quad_s = images_to_primal_quadratics(
            bgr_images=cam1_bgr_images,
            xyz_images=cam1_xyz_images,
            segmenter=cam1_segmenter,
            min_cluster_size=2
        )
        self.base_t_ellipsoid_s = b_t_e_s
        self.primal_quadratic_s = prim_quad_s
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
            visualize_pne_optimisation:bool = False,
            visualize_matching:bool = False
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
            visualize_pne_optimisation = visualize_pne_optimisation,
            visualize_matching = visualize_matching
        )
        return creation_function


    def est_base_t_cam2(self,cam2_bgr_image: np.ndarray, number_retry:int = 2, time_tracker:TimeTracker = TimeTracker()) -> np.ndarray | None:
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
            number_retry=number_retry
        )
        time_tracker.add_time_stamp("Point based initial guess")
        if est_base_t_cam is None:
            return None

        optimized_cam_t_base = self.update_pose(
            cam2_bgr_image=cam2_bgr_image,
            rough_cam_t_base=np.linalg.inv(est_base_t_cam),
            time_tracker=time_tracker
        )
        if optimized_cam_t_base is None:
            print(f"ellipsoid optimisation failed")
            return est_base_t_cam
        
        return np.linalg.inv(optimized_cam_t_base)
    

    def update_pose(self,cam2_bgr_image: np.ndarray, rough_cam_t_base:np.ndarray, time_tracker:TimeTracker) -> np.ndarray | None:
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

        proj_gauss_elli_mu, proj_gauss_elli_sigmas = primal_conics_to_gaussian_ellipses(proj_primal_conics)
        proj_gaussian_ellipses = gauss_ellipse_batch_tuple_to_mat_batch(proj_gauss_elli_mu, proj_gauss_elli_sigmas)

        time_tracker.add_time_stamp("Projecting the 3d ellipsoids to 2d Gauss")

        observed_primal_conics = image_to_primal_conics(
            bgr_image=cam2_bgr_image, 
            segmenter=self.cam2_segmenter,
        )
        time_tracker.add_time_stamp("Image to primal conics")
        if observed_primal_conics.shape[0] == self.min_number_matched_ellipsoids_for_opt:
            return None

        obs_gauss_elli_mu, obs_gauss_elli_sigmas = primal_conics_to_gaussian_ellipses(observed_primal_conics)
        obs_gauss_ellipses = gauss_ellipse_batch_tuple_to_mat_batch(obs_gauss_elli_mu, obs_gauss_elli_sigmas)
        time_tracker.add_time_stamp("Primal conics to gaussians")

        proj_match_idx_s, obs_match_idx_s = match_gaussians_hungarian_on_wasserstein(
            sigma_mu1_s=proj_gaussian_ellipses,
            sigma_mu2_s=obs_gauss_ellipses,
            image_size=cam2_bgr_image.shape[:2],
            config=self.matching_config,
            visualize_matching=cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB) if self.visualize_matching else None
        )
        time_tracker.add_time_stamp("Matching the 2d gaussians")

        if proj_match_idx_s.shape[0] < self.min_number_matched_ellipsoids_for_opt:
            print(f"to few ellipsoids for optimisation: {proj_match_idx_s.shape[0]}")
            return None

        #try:
        cam2_t_base_opt = self.pne_optimizer.optimize_pne(
            initial_cam_t_base=rough_cam_t_base,
            primal_quadratics=self.primal_quadratic_s[proj_match_idx_s],
            primal_conicals= np.array(observed_primal_conics)[obs_match_idx_s],
            intrinsic_cam_mat=scaled_cam2_intrinsics,
            visualize_result= cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB) if self.visualize_pne_optimisation else None
        )
        #except Exception as e:
        #    print(f"Optimisation failed with error: {e} \n\n returning rough pose")
        #    return rough_cam_t_base
        
        time_tracker.add_time_stamp("PNE optimisation")

        return cam2_t_base_opt


if __name__ == "__main__":
    robot_data = RobotEnvironment.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/pose_estimation/out_data_re")
    headset_data = HeadsetData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/pose_estimation/out_data_he")
    
    tt_pne = TimeTracker()
    
    #pne_optimizer = PyposePNEOptimizer(PyposePnEOptimizerConfig(), tt_pne)
    #pne_optimizer = PnEDeltaPoseLBFGSOptimizer(time_tracker=tt_pne)
    pne_optimizer = PnEDeltaPoseAdamOptimizer(time_tracker=tt_pne)

    predictor = EllipsoidPredictor(
        cam2_intrinsic_mtx=headset_data.intrinsic_cam_mtx,
        cam1_bgr_images=robot_data.robot_bgr_images,
        cam1_xyz_images=robot_data.robot_xyz_images,
        extract_and_match_wrapper_config=ExtractAndMatchWrapperConfig(
            rotation_augmentations=[Rotate180Deg],
            extract_and_match=ExtractAndLightGlue(),
            ransac_config=pose_estimation_ransaac_config_less_precise,
        ),
        pne_optimizer=pne_optimizer,
        ellipsoid_refinement_at_res=(1400, 1400),
        cam1_segmenter=SAM3Segmenter(Sam3Prompt()),
        cam2_segmenter=YOLOv26Segmenter("yoloe-26l-seg.pt"),#SAM3Segmenter(Sam3Prompt())
        matching_no_match_cost=0.01
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
    print(f"\n tt_pne")
    tt_pne.print_report()
    #print(f"\n pypose pne:")
    #pne_optimizer.optimize_pne_tt.print_report()
    #print(f"avg number fw calls: {np.mean(pne_optimizer.number_fw_calls)}")

    predictor.extract_and_match_wrapper.print_used_augmentations()
    print(f"avg number of tries: {predictor.extract_and_match_wrapper.avg_number_of_tries()}")