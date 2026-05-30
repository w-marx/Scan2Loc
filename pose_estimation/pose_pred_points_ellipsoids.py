import cv2
import numpy as np
import open3d as o3d

from predictor_handling import *
from extractors_and_matchers import *
from image_augmentation import *
import matplotlib.pyplot as plt
from union_find import UnionFind
from scipy.optimize import linear_sum_assignment
from pne_optimizer import optimize_pne, PnEOptimizerConfig
from sheduler import *


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foreground_segmentation import get_object_masks, Sam3Prompt, display_image_masks

from ellipsoid_utilities_numpy import *

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
        prompt:Sam3Prompt,
        max_centroid_dist: float = 0.03,
        max_color_dist:float = 500,
        visualize_masks:bool = False,
        min_number_points_per_detected_object:int = 1000,
        min_cluster_size:int = 1,
)-> tuple[np.ndarray, np.ndarray]:
    """
    Generates M ellipsoids from an environment
    :param bgr_images: An NxHxWx3-uint8 array of BGR images
    :param xyz_images: An NxHxWx3-float array of 3d points in the base frame
    :param prompt: The Sam3Prompt to detect objects in an image
    :param max_centroid_dist: The maximum distance in meters between two object centers in different images to be considered the same
    :return: the base_t_ellipsoid hom. matrices (Mx4x4) and the primal quadratics (Mx4x4)
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
        print(f"started mask generation")
        object_masks = get_object_masks(bgr_img, prompt)
        for object_mask in object_masks:
            pc = xyz_img[object_mask > 0]
            if pc.shape[0] < min_number_points_per_detected_object:
                continue
            mask_storage.add_new_mask(object_mask)
            object_avg_colors.append(np.mean(bgr_img[object_mask > 0], axis = 0))
            object_avg_centers.append(np.mean(pc, axis = 0))
            object_image_idxs.append(i)
        if visualize_masks:
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
    
    visualize_primal_quadratics(
        base_t_ellipsoid_s=fused_base_t_ellipsoid_s,
        primal_quadratic_s=fused_primal_quaddratic_s,
        bg_point_cloud=xyz_images.reshape(-1,3),
        bg_point_cloud_colors=bgr_images.reshape(-1,3)
    )

    return fused_base_t_ellipsoid_s, fused_primal_quaddratic_s

def image_to_primal_conics(bgr_image:np.ndarray, sam3_prompt:Sam3Prompt) -> np.ndarray:
    """
    Creates a Nx3x3 batch of primal conics from the image by segmenting it using sam3.
    :param bgr_image: the bgr image
    :param sam3_prompt: the sam3 prompt config
    :return: the primal conics (Nx3x3)
    """
    assert assert_mxnx3_np_uint8_image(bgr_image)

    object_masks = get_object_masks(bgr_image, sam3_prompt)
    primal_conic_s = []

    for object_mask in object_masks:
        rows, cols = np.where(object_mask)
        pc_2d = np.column_stack((cols, rows))
        primal_conic_s.append(fit_primal_conic_to_2d_point_cloud(pc_2d))

    return np.array(primal_conic_s)


def match_gaussians(sigma_mu1_s, sigma_mu2_s):
    """
    
    """
    n, m = sigma_mu1_s.shape[0], sigma_mu2_s.shape[0]
    assert all([assert_gaussian_ellipse_mat(sigma_mu) for sigma_mu in sigma_mu1_s])
    assert all([assert_gaussian_ellipse_mat(sigma_mu) for sigma_mu in sigma_mu2_s])

    adjecency_mat = np.full((max(n,m), max(n,m)), 1e12)

    for i1, sigma_mu1 in enumerate(sigma_mu1_s):
        for i2, sigma_mu2 in enumerate(sigma_mu2_s):
            adjecency_mat[i1, i2] = wasserstein_distance_sq(sigma_mu1, sigma_mu2)

    # TODO use batch

    row_ind, col_ind = linear_sum_assignment(adjecency_mat)

    valid_matches = (row_ind < n) & (col_ind < m)

    return row_ind[valid_matches], col_ind[valid_matches]




class EllipsoidPredictor(PosePredictor):
    def __init__(
            self,
            cam2_intrinsic_mtx:np.ndarray,
            cam1_bgr_images:np.ndarray,
            cam1_xyz_images:np.ndarray,
            extract_and_match:ExtractAndMatch = ExtractAndMatchLoMa(),
            pne_config:PnEOptimizerConfig = PnEOptimizerConfig(),
            ransac_config:RansacPoseEstimationConfig = pose_estimation_ransaac_config_precise,
            sheduler:Sheduler = EMASheduler,
            number_tries_b4_giving_up:int = 1,
        ):
        super().__init__()
        b_t_e_s, prim_quad_s = images_to_primal_quadratics(
            bgr_images=cam1_bgr_images,
            xyz_images=cam1_xyz_images,
            prompt=Sam3Prompt(),
            min_cluster_size=2
        )
        self.cam1_bgr_images = cam1_bgr_images
        self.cam1_xyz_images = cam1_xyz_images
        self.base_t_ellipsoid_s = b_t_e_s
        self.primal_quadratic_s = prim_quad_s
        self.pne_optimizer_config = pne_config
        self.cam2_intrinsic_mtx = cam2_intrinsic_mtx
        self.extract_and_match = extract_and_match
        self.initial_raansac_guess_config = ransac_config
        self.sheduler = sheduler(cam1_bgr_images.shape[0])
        self.number_tries_b4_giving_up = number_tries_b4_giving_up


    def est_base_t_cam2(self,cam2_bgr_image: np.ndarray,time_tracker:TimeTracker) -> np.ndarray | None:
        """
        Predicts the homogenous transformation base_t_cam2
        """
        number_tries = 0
        est_base_t_cam = None
        while est_base_t_cam is None and number_tries < self.number_tries_b4_giving_up:
            idx = self.sheduler.get_best()
            est_base_t_cam = self.est_base_t_cam2_helper(
                cam1_bgr_image = self.cam1_bgr_images[idx],
                base_xyz_image = self.cam1_xyz_images[idx],
                cam2_bgr_image = cam2_bgr_image,
                time_tracker = time_tracker
            )
            self.sheduler.adjust(idx, est_base_t_cam is not None)
            number_tries += 1
        if est_base_t_cam is not None:
            return np.linalg.inv(self.update_pose(
                cam2_bgr_image=cam2_bgr_image,
                rough_cam_t_base=np.linalg.inv(est_base_t_cam),
                time_tracker=time_tracker
            ))
        return None
    
    def update_pose(self,cam2_bgr_image: np.ndarray, rough_cam_t_base:np.ndarray, time_tracker:TimeTracker) -> np.ndarray | None:
        # TODO add some fallback if matching doesnt work out (recenter with point matching)

        time_tracker.reset_elapsed_time()
        proj_primal_conics = project_primal_quadratics_to_primal_conicals(
            primal_quadratics= self.primal_quadratic_s,
            cam_t_base=rough_cam_t_base,
            intrinsic_mtx=self.cam2_intrinsic_mtx,
        )
        proj_gauss_elli_mu, proj_gauss_elli_sigmas = primal_conics_to_gaussian_ellipses(proj_primal_conics)
        proj_gaussian_ellipses = gauss_ellipse_batch_tuple_to_mat_batch(proj_gauss_elli_mu, proj_gauss_elli_sigmas)

        observed_primal_conics = image_to_primal_conics(bgr_image=cam2_bgr_image, sam3_prompt=Sam3Prompt())

        obs_gauss_elli_mu, obs_gauss_elli_sigmas = primal_conics_to_gaussian_ellipses(observed_primal_conics)
        obs_gauss_ellipses = gauss_ellipse_batch_tuple_to_mat_batch(obs_gauss_elli_mu, obs_gauss_elli_sigmas)

        time_tracker.add_time_stamp("Projecting for matching")
        proj_match_idxs, obs_match_idxs = match_gaussians(proj_gaussian_ellipses, obs_gauss_ellipses)
        time_tracker.add_time_stamp("Matching")

        cam2_t_base_opt = optimize_pne(
            initial_cam_t_base=rough_cam_t_base,
            primal_quadratics=self.primal_quadratic_s[proj_match_idxs],
            primal_conicals=np.array(observed_primal_conics)[obs_match_idxs],
            intrinsic_cam_mat=self.cam2_intrinsic_mtx,
            config=self.pne_optimizer_config,
            visualize_result=None#cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)
        )
        time_tracker.add_time_stamp("PNE optimisation")
        return cam2_t_base_opt


    
    def est_base_t_cam2_helper(self,
                            cam1_bgr_image:np.ndarray,
                            base_xyz_image:np.ndarray,
                            cam2_bgr_image: np.ndarray,
                            time_tracker:TimeTracker
        ):
        
        time_tracker.reset_elapsed_time()
        
        cam1_rgb_image = cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2RGB)
        cam2_rgb_image = cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)

        image_points_cam1, image_points_cam2 = self.extract_and_match.get_matched_points(
            img1_rgb=cam1_rgb_image, img2_rgb=cam2_rgb_image, plot_results = False
        )
        time_tracker.add_time_stamp("Extract and Match")
        world_obj_points = np.array([base_xyz_image[int(np.round(y)),int(np.round(x))] for x,y in image_points_cam1])

        cam2_t_base_pnp__inliers = estimate_point_pose_ransac(
            img_points=image_points_cam2, 
            world_points=world_obj_points, 
            intrinsic_matrix=self.cam2_intrinsic_mtx, 
            config=self.initial_raansac_guess_config
        )
        if cam2_t_base_pnp__inliers is None:
            return None
        cam2_t_base_pnp, inliers = cam2_t_base_pnp__inliers
    
        return np.linalg.inv(cam2_t_base_pnp)


if __name__ == "__main__":
    robot_data = RobotEnvironment.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data_r")
    headset_data = HeadsetData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data_h")
    predictor = EllipsoidPredictor(
        cam2_intrinsic_mtx=headset_data.intrinsic_cam_mtx,
        cam1_bgr_images=robot_data.robot_bgr_images,
        cam1_xyz_images=robot_data.robot_xyz_images,
        extract_and_match=ExtractAndLightGlue(),
    )

    tt1 = TimeTracker()
    tt2 = TimeTracker()
    grader = OnePredictorRecordingGrader(
        predictor=predictor, 
        headset_data=headset_data,
        prediction_time_tracker=tt1,
        subcomponent_time_tracker=tt2
    )
    
    print(f"translat errors: \n {grader.translational_errors()} \n")
    #grader.visualize_predictions()
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