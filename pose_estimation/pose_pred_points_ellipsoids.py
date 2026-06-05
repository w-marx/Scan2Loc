import cv2
import numpy as np
import open3d as o3d

from predictor_handling import *
from extractors_and_matchers import *
from image_augmentation import *
import matplotlib.pyplot as plt
from union_find import UnionFind
from scipy.optimize import linear_sum_assignment
from pypose_pne_optimizer import *
from sheduler import *
import matplotlib.lines as mlines


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foreground_segmentation import *

from ellipsoid_utilities_numpy import *
from pne_delta_pose_otimizer import *

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

def match_gaussians_hungarian_on_wasserstein(
        sigma_mu1_s:np.ndarray, 
        sigma_mu2_s:np.ndarray,
        image_size:tuple[int, int],
        dummy_value:float = 1,
        k:float = 10,
        visualize_matching:None | np.ndarray = None
    ):
    """
    Uses the hungarian algorithm to match distributions
    :param sigma_mu1_s: Nx2x3 array of the form [[sigma_1 | mu_1], ... ]
    :param sigma_mu2_s: Mx2x3 array of the form [[sigma_1 | mu_1], ... ]
    :param image_size: A tuple of the image size (h,w) where the gaussians come from (used for normalisation)
    :param dummy_value: The value for the no-match alternative
    :param k: valid matches hav the cost: cost < median-cost + k * mad
    """
    n, m = sigma_mu1_s.shape[0], sigma_mu2_s.shape[0]
    assert assert_gaussian_ellipse_mat_batch(sigma_mu1_s)
    assert assert_gaussian_ellipse_mat_batch(sigma_mu2_s)

    h, w = image_size
    sigma_mu1_s_n = normalize_gaussians(sigma_mu1_s, h, w)
    sigma_mu2_s_n = normalize_gaussians(sigma_mu2_s, h, w)

    adjecency_mat = np.full((m+n, m+n), dummy_value, dtype = np.float32)
    mu1_s, sigma1_s = gauss_ellipse_mat_batch_to_tuple(sigma_mu1_s_n)
    mu2_s, sigma2_s = gauss_ellipse_mat_batch_to_tuple(sigma_mu2_s_n)
    adjecency_mat[:n, :m] = pairwise_sq_wasserstein_distance(mu1_s, sigma1_s, mu2_s, sigma2_s)

    row_ind, col_ind = linear_sum_assignment(adjecency_mat)
    valid_ind = (col_ind < m) & (row_ind < n)
    row_ind, col_ind = row_ind[valid_ind], col_ind[valid_ind]
    
    median_cost = np.median(adjecency_mat[row_ind, col_ind])
    mad = np.median(np.abs(adjecency_mat[row_ind, col_ind] - median_cost))

    valid_matches = (adjecency_mat[row_ind, col_ind] < median_cost+ k * mad)
    matched_idx_1_s, matched_idx_2_s = row_ind[valid_matches], col_ind[valid_matches]

    if visualize_matching is not None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plt.imshow(visualize_matching, extent=[0, 1, 0, 1], origin="lower")
        ax.set_ylim(1, 0)

        elli1_s = gaussian_ellipse_s_to_matplotlib_ellipse_s(sigma_mu1_s_n, line_style="--", line_widths=2, colors='black')
        elli2_s = gaussian_ellipse_s_to_matplotlib_ellipse_s(sigma_mu2_s_n, line_style="-", line_widths=2, colors='black')

        ax.set_title("Matching visualisation")

        for e1 in elli1_s:
            ax.add_patch(e1)
        for e2 in elli2_s:
            ax.add_patch(e2)

        ax.scatter(sigma_mu1_s_n[:,0,2],sigma_mu1_s_n[:,1,2], s=5, marker = 'o', color = 'black')
        ax.scatter(sigma_mu2_s_n[:,0,2],sigma_mu2_s_n[:,1,2], s=5, marker = 's', color = 'black')

        ax.quiver(
            sigma_mu1_s_n[matched_idx_1_s,0,2],
            sigma_mu1_s_n[matched_idx_1_s,1,2],
            sigma_mu2_s_n[matched_idx_2_s,0,2]-sigma_mu1_s_n[matched_idx_1_s,0,2], 
            sigma_mu2_s_n[matched_idx_2_s,1,2]-sigma_mu1_s_n[matched_idx_1_s,1,2],
            angles='xy', scale_units='xy', scale=1,
            color='lime',
            alpha=0.6,
            width=0.005
        )

        empty_lines = [
            mlines.Line2D([], [], color='black', linestyle='-',  linewidth=2, label='Frame 1'),
            mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Frame 2'),
        ]
        ax.legend(handles=empty_lines, loc='upper right')
        plt.show()
    return matched_idx_1_s, matched_idx_2_s    


class EllipsoidPredictor(PosePredictor):
    def __init__(
            self,
            cam2_intrinsic_mtx:np.ndarray,
            cam1_bgr_images:np.ndarray,
            cam1_xyz_images:np.ndarray,
            extract_and_match_wrapper_config:ExtractAndMatchWrapperConfig,
            cam1_segmenter:Segmenter = SAM3Segmenter(Sam3Prompt()),
            cam2_segmenter:Segmenter = YOLOv26Segmenter(),
            pne_optimizer:PnEOptimizer = PyposePNEOptimizer(),
            time_tracker_init:TimeTracker = TimeTracker(),
            min_number_matched_ellipsoids_for_opt = 1,
            ellipsoid_refinement_at_res: None | tuple[int, int] = None,
            matching_no_match_cost = 0.05,
            matching_mad_dist = 1000
        ):
        super().__init__()
        self.cam2_intrinsic_mtx = cam2_intrinsic_mtx
        self.pne_optimizer = pne_optimizer
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
        self.cam1_segmenter = cam1_segmenter
        self.cam2_segmenter = cam2_segmenter

        self.matching_no_match_cost = matching_no_match_cost
        self.matching_mad_k = matching_mad_dist


    def est_base_t_cam2(self,cam2_bgr_image: np.ndarray, number_retry:int = 2, time_tracker:TimeTracker = TimeTracker()) -> np.ndarray | None:
        """
        Predicts the homogenous transformation base_t_cam2
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
            dummy_value=self.matching_no_match_cost,
            k = self.matching_mad_k,
            visualize_matching=None#cam2_bgr_image
        )
        time_tracker.add_time_stamp("Matching the 2d gaussians")

        if proj_match_idx_s.shape[0] < self.min_number_matched_ellipsoids_for_opt:
            print(f"to few ellipsoids for optimisation: {proj_match_idx_s.shape[0]}")
            return None

        #try:
        cam2_t_base_opt = self.pne_optimizer.optimize_pne(
            initial_cam_t_base=rough_cam_t_base,
            primal_quadratics=self.primal_quadratic_s[proj_match_idx_s],
            primal_conicals=np.array(observed_primal_conics)[obs_match_idx_s],
            intrinsic_cam_mat=scaled_cam2_intrinsics,
            visualize_result=None#cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)
        )
        #except Exception as e:
        #    print(f"Optimisation failed with error: {e} \n\n returning rough pose")
        #    return rough_cam_t_base
        
        time_tracker.add_time_stamp("PNE optimisation")

        print(f"optimisation difference: {np.sum(np.abs(rough_cam_t_base-cam2_t_base_opt))}")
        return cam2_t_base_opt


if __name__ == "__main__":
    robot_data = RobotEnvironment.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data_r")
    headset_data = HeadsetData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data_h")
    
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