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


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foreground_segmentation import get_object_masks, Sam3Prompt

from ellipsoid_utilities import * 


def images_to_objects(
        bgr_images:np.ndarray,
        xyz_images:np.ndarray,
        prompt:Sam3Prompt,
        max_centroid_dist: float = 0.05,
        debug_vis_3d:bool = False,
        visualize_masks:bool = False,
        min_number_points_per_detected_object:int = 1000,
        min_cluster_size:int = 2
)-> tuple[np.ndarray, np.ndarray]:
    """
    Generates M ellipsoids from an environment
    :param bgr_images: An NxHxWx3-uint8 array of BGR images
    :param xyz_images: An NxHxWx3-float array of 3d points in the base frame
    :param prompt: The Sam3Prompt to detect objects in an image
    :param max_centroid_dist: The maximum distance in meters between two object centers in different images to be considered the same
    :param debug_vis_3d: Whether to visualize the objects as point-clouds or not
    :return: the base_t_ellipsoid hom. matrices (Mx4x4) and the primal quadratics (Mx4x4)
    """
    assert all([assert_mxnx3_np_uint8_image(bgr_img) for bgr_img in bgr_images])
    assert  xyz_images.ndim == 4 and xyz_images.shape[-1] == 3
    assert  bgr_images.shape[:3] == xyz_images.shape[:3]

    # Generate objects

    # (n-all-objs)xHxW-bool masks
    object_avg_colors = []
    primal_quaddratics = []
    base_t_ellipsoid_s = []

    for bgr_img, xyz_img in zip(bgr_images, xyz_images):
        print(f"started mask generation")
        object_masks = get_object_masks(bgr_img, prompt)
        for object_mask in object_masks:
            pc = xyz_img[object_mask > 0]
            if pc.shape[0] < min_number_points_per_detected_object:
                continue
            object_avg_colors.append(np.mean(bgr_img[object_mask], axis = 0))
            base_t_ellipsoid, primal_quadratic = fit_ellipsoid_to_3d_point_cloud(pc, visualize=False)
            base_t_ellipsoid_s.append(base_t_ellipsoid)
            primal_quaddratics.append(primal_quadratic)
        if visualize_masks:
            plt.figure(figsize=(15, 5))
            plt.imshow(bgr_img)
            for obj_mask in object_masks:
                plt.imshow(obj_mask, alpha=0.4, cmap='jet')
            plt.show()
    
    object_avg_colors = np.stack(object_avg_colors, axis = 0)
    base_t_ellipsoid_s = np.stack(base_t_ellipsoid_s, axis = 0)
    primal_quaddratics = np.stack(primal_quaddratics, axis = 0)

    visualize_primal_quadratics(
        base_t_ellipsoids=base_t_ellipsoid_s,
        primal_quadratics=primal_quaddratics
    )

    # Join similar objects
    n = object_avg_colors.shape[0]
    print(f"number of objects")

    # NxN adjecency matrix
    centroid_distance_matrix = np.linalg.norm(
        base_t_ellipsoid_s[:, :3, 3][:, None] - base_t_ellipsoid_s[:, :3, 3][None, :], axis=-1
    )
    centroid_distance_matrix_mask = centroid_distance_matrix < max_centroid_dist

    union_find = UnionFind(n)
    for row_idx in range(n):
        for col_idx in range(row_idx+1, n):
            if centroid_distance_matrix_mask[row_idx, col_idx]:
                union_find.union(row_idx, col_idx)

    fused_base_t_ellipsoid_s = []
    fused_primal_quaddratic_s = []
    for i,cluster in enumerate(union_find.return_clusters()):
        if len(cluster) < min_cluster_size:
            continue
        b_t_e, p_q = fuse_ellipsoids(base_t_ellipsoid_s[cluster], primal_quaddratics[cluster])
        fused_base_t_ellipsoid_s.append(b_t_e)
        fused_primal_quaddratic_s.append(p_q)
    fused_base_t_ellipsoid_s = np.array(fused_base_t_ellipsoid_s)
    fused_primal_quaddratic_s = np.array(fused_primal_quaddratic_s)
    
    visualize_primal_quadratics(
        base_t_ellipsoids=fused_base_t_ellipsoid_s,
        primal_quadratics=fused_primal_quaddratic_s,
        bg_point_cloud=xyz_images.reshape(-1,3),
        bg_point_cloud_colors=bgr_images.reshape(-1,3)
    )

    return fused_base_t_ellipsoid_s, fused_primal_quaddratic_s

def image_to_primal_conics(bgr_image:np.ndarray, sam3_prompt:Sam3Prompt):
    object_masks = get_object_masks(bgr_image, sam3_prompt)
    base_t_ellipsoid_s = []
    primal_conic_s = []

    for object_mask in object_masks:
        rows, cols = np.where(object_mask)
        pc_2d = np.column_stack((cols, rows))
        base_t_ellipsoid, primal_conic = fit_primal_conic_to_2d_point_cloud(pc_2d)
        base_t_ellipsoid_s.append(base_t_ellipsoid)
        primal_conic_s.append(primal_conic)

    return base_t_ellipsoid_s, primal_conic_s

def match_gaussians(sigma_mu1_s, sigma_mu2_s):
    """
    
    """
    n, m = sigma_mu1_s.shape[0], sigma_mu2_s.shape[0]
    assert all([assert_gaussian_ellipse_mat(sigma_mu) for sigma_mu in sigma_mu1_s])
    assert all([assert_gaussian_ellipse_mat(sigma_mu) for sigma_mu in sigma_mu2_s])

    adjecency_mat = np.full((max(n,m), max(n,m)), 1)

    for i1, sigma_mu1 in enumerate(sigma_mu1_s):
        for i2, sigma_mu2 in enumerate(sigma_mu2_s):
            adjecency_mat[i1, i2] = wasserstein_distance_sq(sigma_mu1, sigma_mu2)
    

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
        ):
        super().__init__()
        self.base_ellipsoid_s, self.quadratic_ellipsoid_s = images_to_objects(
            bgr_images=cam1_bgr_images[:1],
            xyz_images=cam1_xyz_images[:1],
            prompt=Sam3Prompt(),
            debug_vis_3d=False,
            min_cluster_size=1
        )
        self.pne_optimizer_config = pne_config
        self.cam2_intrinsic_mtx = cam2_intrinsic_mtx
        self.extract_and_match = extract_and_match
        self.initial_raansac_guess_config = ransac_config


    def improve_pose(self, 
                    cam2_t_base_init:np.ndarray,
                    cam2_bgr_image:np.ndarray
                    ):
        
        proj_primal_conics = [
            project_primal_quadratic_to_primal_conical(pq,cam2_t_base_init,self.cam2_intrinsic_mtx) 
            for pq in self.quadratic_ellipsoid_s
        ]
        proj_gaussian_ellipses = np.array([primal_conic_to_gaussian_ellipse(pc) for pc in proj_primal_conics])

        base_t_obs_ellipses, observed_primal_conics = image_to_primal_conics(bgr_image=cam2_bgr_image, sam3_prompt=Sam3Prompt())
        observed_gaussian_ellipses = np.array([primal_conic_to_gaussian_ellipse(pc) for pc in observed_primal_conics])

        proj_match_idxs, obs_match_idxs = match_gaussians(proj_gaussian_ellipses, observed_gaussian_ellipses)

        proj_matplotlib_ellipses = gaussian_ellipse_s_to_matplotlib_ellipse_s(proj_gaussian_ellipses)
        observed_matplotlib_ellipses = gaussian_ellipse_s_to_matplotlib_ellipse_s(observed_gaussian_ellipses, colors="red")

        proj_matplotlib_ellipses_m = gaussian_ellipse_s_to_matplotlib_ellipse_s(proj_gaussian_ellipses[proj_match_idxs], line_widths=3)
        observed_matplotlib_ellipses_m = gaussian_ellipse_s_to_matplotlib_ellipse_s(observed_gaussian_ellipses[obs_match_idxs], colors="red", line_widths=3)

        plot_ellipses(
            bgr_img=cam2_bgr_image,
            ellipses=proj_matplotlib_ellipses+observed_matplotlib_ellipses+proj_matplotlib_ellipses_m+observed_matplotlib_ellipses_m
        )
        print(f"obs_match_idx: {obs_match_idxs}")

        cam2_t_base_opt = optimize_pne(
            initial_cam_t_base=cam2_t_base_init,
            primal_quadratics=self.quadratic_ellipsoid_s[proj_match_idxs],
            primal_conicals=np.array(observed_primal_conics)[obs_match_idxs],
            intrinsic_cam_mat=self.cam2_intrinsic_mtx,
            config=self.pne_optimizer_config,
            visualize_result=cam2_bgr_image
        )
        print(f"optimized: {cam2_t_base_opt}") 
    

    def est_base_t_cam2(self,
                        cam1_bgr_image:np.ndarray,
                        base_xyz_image:np.ndarray,
                        cam2_bgr_image: np.ndarray,
                        time_tracker:TimeTracker
                        ) -> np.ndarray | None:
        
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

        self.improve_pose(cam2_t_base_pnp, cam2_bgr_image)
    
        return np.linalg.inv(cam2_t_base_pnp)

if __name__ == "__main__":
    data = PredictionData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data")
    predictor = EllipsoidPredictor(
        cam2_intrinsic_mtx=data.headset_intrinsics,
        cam1_bgr_images=data.robot_bgr_images,
        cam1_xyz_images=data.robot_xyz_images,
        extract_and_match=ExtractAndLightGlue(),
        ransac_config=pose_estimation_ransaac_config_precise
    )
    grader = OnePredictorOneDatasetGrader(predictor=predictor, data=data)
    
    #grader.visualize_predictions()
    
    print(f"median rot error: {np.round(np.rad2deg(grader.median_rotational_error()), 2)} degrees")
    print(f"median translational error: {np.round(grader.median_translational_error()*1000, 1)} mm")
    print(f"avg. sub median rot error: {np.round(np.rad2deg(grader.average_sub_median_rotational_error()), 2)} degrees")
    print(f"avg. sub median translational error: {np.round(grader.average_sub_median_translat_error()*1000, 1)} mm")
    print(f"sucess_ratio: {np.round(grader.sucess_ratio(),2)}")

    #tt = TimeTracker()
    #for i in range(10):
    #    grader = OnePredictorOneDatasetGrader(predictor=predictor, data=data, time_tracker=tt)
    #tt.print_report()