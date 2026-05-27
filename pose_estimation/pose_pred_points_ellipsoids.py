import cv2
import numpy as np
import open3d as o3d

from predictor_handling import *
from extractors_and_matchers import *
from image_augmentation import *
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from union_find import UnionFind

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from foreground_segmentation import get_object_masks, Sam3Prompt

def assert_primal_conical_hom_ellipse(primal_conical_hom:np.ndarray)->bool:
    """
    Raises an assertion error if the primal quadratic is infeasible
    Assumed is the form:
    | A     B/2     D   |
    | B/2   C       E/2 |
    | D     E/2     F   |
    Where points are on the ellipse if:
    Ax + Bxy + Cx + Dx + Ey + F = 0

    :param primal_conical_hom: The matrix to be checked
    return True
    """
    det_a_33 = np.linalg.det(primal_conical_hom[:2, :2])
    det_a_q = np.linalg.det(primal_conical_hom)

    assert det_a_q == 0, "Ellipse is degenerate"
    assert det_a_33 > 0, f"Primal quadratic is not an ellipse: 0 < det(A_33) = {det_a_33}"
    assert det_a_33*det_a_q < 0, f"Ellipse is not real"
    assert np.allclose(primal_conical_hom, primal_conical_hom.T), "Matrix must be symmetric"
    return True


def images_to_objects(
        bgr_images:np.ndarray,
        xyz_images:np.ndarray,
        prompt:Sam3Prompt,
        max_centroid_dist: float = 0.05,
        debug_vis_3d:bool = False
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
    assert_mxnx3_np_uint8_image(bgr_images)
    assert  xyz_images.ndim == 4 and xyz_images.shape[-1] == 3
    assert  bgr_images.shape[:3] == xyz_images.shape[:3]

    # Generate objects

    # (n-all-objs)xHxW-bool masks
    all_object_masks = []
    all_object_point_clouds = []

    for bgr_img, xyz_img in zip(bgr_images, xyz_images):
        object_masks = get_object_masks(bgr_img, prompt)
        all_object_masks += object_masks
        all_object_point_clouds += [xyz_img[mask] for mask in object_masks]

    all_object_masks = np.array(all_object_masks)
    n = all_object_masks.shape[0]
    all_object_point_clouds = np.array(all_object_point_clouds)
    all_object_means = np.mean(all_object_point_clouds, axis=1)

    # Join similar objects

    # NxN adjecency matrix
    centroid_distance_matrix = np.linalg.norm(
        all_object_means[:, None] - all_object_means[None, :], axis=-1
    )
    centroid_distance_matrix_mask = centroid_distance_matrix < max_centroid_dist

    union_find = UnionFind(n)
    all_indices = np.arange(0,n)

    for row_idx, row in enumerate(centroid_distance_matrix_mask):
        for neighbor in all_indices[row]:
            union_find.union(neighbor, row_idx)

    # Generate cluster point_clouds
    point_clouds = []
    for cluster in union_find.return_clusters():
        point_clouds.append(np.concatenate(all_object_point_clouds[cluster]))

    # Generate ellipsoids
    base_t_ellipsoid_s = []
    primal_quadratic_s = []

    if debug_vis_3d:
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

        to_vis = [base_frame]

        for pc in point_clouds:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            pcd.paint_uniform_color([0, 0, 0])
            to_vis.append(pcd)

        o3d.visualization.draw_geometries(to_vis, f"3D features visualization")

    for point_cloud in point_clouds:
        base_t_ellipsoid, primal_quadratic = fit_ellipsoid_to_3d_point_cloud(point_cloud)
        base_t_ellipsoid_s.append(base_t_ellipsoid)
        primal_quadratic_s.append(primal_quadratic)

    return np.array(base_t_ellipsoid_s), np.array(primal_quadratic_s)


def fit_ellipsoid_to_3d_point_cloud(point_cloud:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    :param point_cloud: An Nx3 point cloud in the base_frame
    :return a tuple of the base_t_ellipsoid hom. mtx and the primal quadratic
    """
    assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 3

    # Get center
    center = point_cloud.mean(axis=0)  # (3,)
    pts_centered = point_cloud - center


    cov_matrix = np.cov(pts_centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]

    base_t_ellipsoid = r_t_to_hom(eigvecs[:, idx], center)

    # to local orientation
    pts_local = (base_t_ellipsoid[:3, :3] @ pts_centered.T).T  # (N,3)

    # a b theta in local coordinate
    a = np.max(np.abs(pts_local[:, 0]))
    b = np.max(np.abs(pts_local[:, 1]))
    c_axis = np.max(np.abs(pts_local[:, 2]))
    # TODO replace with something more robust

    # quadric in world coordinate
    q_local = np.diag([1/a**2, 1/b**2, 1/c_axis**2, -1.0])

    ellipsoid_t_base = np.linalg.inv(base_t_ellipsoid)
    primal_quadratic = ellipsoid_t_base.T @ q_local @ ellipsoid_t_base

    return base_t_ellipsoid, primal_quadratic

def fit_ellipsoid_to_2d_point_cloud(point_cloud:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    :param point_cloud: An Nx2 point cloud
    :return a tuple of the base_t_ellipsoid hom. mtx and the primal quadratic
    """
    assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 2

    base_center = np.mean(point_cloud, axis=0)
    base_pts_centered = point_cloud - base_center
    cov_matrix = np.cov(base_pts_centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]

    base_t_ellipsoid = r_t_to_hom(eigvecs[:, idx], base_center)

    ellipsoid_pts_centered = (base_t_ellipsoid[:2, :2].T @ base_pts_centered.T).T
    a = np.max(np.abs(ellipsoid_pts_centered[:, 0]))
    b = np.max(np.abs(ellipsoid_pts_centered[:, 1]))

    ellipsoid_t_base = np.linalg.inv(base_t_ellipsoid)
    primal_conic = ellipsoid_t_base.T @ np.diag([1/a**2, 1/b**2, -1.0]) @ ellipsoid_t_base

    return base_t_ellipsoid, primal_conic

def project_3d_ellipsoid_to_img_coordinates(
        primal_quadratic:np.ndarray,
        cam_t_base:np.ndarray,
        intrinsic_mtx:np.ndarray
):
    """
    Takes a 4x4 primal quadratic in the world frame
    and returns the 3x3 primal conic in the camera frame
    """
    _ = assert_homogeneous_mat(cam_t_base)
    _ = assert_intrinsic_mat(intrinsic_mtx)

    dual_quadratic = np.linalg.inv(primal_quadratic)
    cam_dual_conic = (intrinsic_mtx @ cam_t_base[:3,:]) @ dual_quadratic @ (intrinsic_mtx @ cam_t_base[:3,:]).T
    cam_primal_conic = np.linalg.inv(cam_dual_conic)

    return cam_primal_conic


def primal_conic_to_gaussian_ellipse(primal_conic:np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    """
    Takes a 3x3 primal conic and returns a normal distribution where
    :param primal_conic: The 3x3 primal conic
    :return a (2,) vector of the middle and a (2x2) covariance matrix
    """
    assert_primal_conical_hom_ellipse(primal_conic)

    a = primal_conic[0:2, 0:2]
    b = primal_conic[0:2, 2]
    c = primal_conic[2, 2]
    mu = -np.linalg.inv(a) @ b

    # normalization
    s = mu.T @ a @ mu - c
    c_norm = primal_conic / s
    a_norm = c_norm[0:2, 0:2]

    sigma = np.linalg.inv(a_norm)

    return mu, sigma



def gaussian_ellipse_s_to_matplotlib_ellipse_s(
        gaussian_ellipse_s:list[tuple[np.ndarray, np.ndarray]],
        colors:str = 'green',
        line_widths:list[int]|int = 1
):
    ellipses = []
    for (mu, sigma) in gaussian_ellipse_s:
        vals, vecs = np.linalg.eigh(sigma)

        # Sort by eigenvalue size
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Ellipse axes lengths = sqrt(eigenvalues)
        width = 2 * np.sqrt(vals[0])
        height = 2 * np.sqrt(vals[1])

        # Orientation angle in degrees
        angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))

        # Create ellipse patch
        ellipse = Ellipse(
            xy=(mu[0], mu[1]),
            width=width,
            height=height,
            angle=angle,
            edgecolor=colors,
            facecolor='none',
            linewidth=line_widths
        )
        ellipses.append(ellipse)
    return ellipses

def plot_ellipses(
        bgr_img:np.ndarray,
        ellipses:list[Ellipse]
):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(rgb_img)

    for ellipse in ellipses:
        ax.add_patch(ellipse)

    ax.set_title(f'Detected Ellipses: {len(ellipses)}')
    ax.axis('off')
    plt.tight_layout()
    plt.show()


class EllipsoidPredictor(PosePredictor):
    def __init__(
            self,
            cam2_intrinsic_mtx:np.ndarray,
            cam1_bgr_images:np.ndarray,
            cam1_xyz_images:np.ndarray,
            extract_and_match:ExtractAndMatch = ExtractAndMatchLoMa(),
            ransac_config:RansacPoseEstimationConfig = pose_estimation_ransaac_config_precise,
            debug_visualize_init_result:bool = True
        ):
        super().__init__()
        self.base_ellipsoid_s, self.quadratic_ellipsoid_s = images_to_objects(
            bgr_images=cam1_bgr_images,
            xyz_images=cam1_xyz_images,
            prompt=Sam3Prompt(),
            max_centroid_dist=0.5
        )

        self.cam2_intrinsic_mtx = cam2_intrinsic_mtx
        self.extract_and_match = extract_and_match
        self.initial_raansac_guess_config = ransac_config
    
    def visualize_3d(self):
        pcd_s = [] # self.object_segmented_point_cloud

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

        print(f"number of objects: {len(pcd_s)}")

        colors = plt.cm.jet(np.linspace(0,1, len(pcd_s)))
        to_vis = [base_frame]
        for i, pcd_np in enumerate(pcd_s):
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pcd_np)
            pcd.paint_uniform_color(colors[i][:3])
            to_vis.append(pcd)

        o3d.visualization.draw_geometries(to_vis, f"3D features visualization")

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

        quadratic_conic_to_gauss = lambda x: (
            primal_conic_to_gaussian_ellipse(
                primal_conic=project_3d_ellipsoid_to_img_coordinates(
                    primal_quadratic=x,
                    cam_t_base=cam2_t_base_pnp,
                    intrinsic_mtx=self.cam2_intrinsic_mtx,
                )
            )
        )

        plot_ellipses(
            bgr_img=cam2_bgr_image,
            ellipses=gaussian_ellipse_s_to_matplotlib_ellipse_s([quadratic_conic_to_gauss(x) for x in self.quadratic_ellipsoid_s])
        )
    
        return np.linalg.inv(cam2_t_base_pnp)




if __name__ == "__main__":
    data = PredictionData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data")
    predictor = EllipsoidPredictor(
        data.headset_intrinsics, 
        point_cloud=data.point_cloud,
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