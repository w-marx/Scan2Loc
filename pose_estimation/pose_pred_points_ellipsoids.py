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

def remove_outliers_from_point_cloud(points:np.ndarray, contamination:float = 0.05)->np.ndarray:
    """
    Uses I-Forest to remove points deemed as outliers
    :param contamination: The percentage of points to remove
    :param points: A Nx3-float numpy array of x,y,z points
    :return: A Mx3-float numpy array of x,y,z points with M <= N
    """
    assert 0 <= contamination <= 1.0
    assert points.ndim == 2 and points.shape[1] == 3

    if contamination == 0:
        return points
    if contamination == 1.0:
        return np.empty((0,3))

    from sklearn.ensemble import IsolationForest
    forest = IsolationForest(contamination=contamination)
    forest.fit(points)
    prediction = forest.predict(points)
    return points[prediction==1]

def assert_primal_conical_hom_ellipse(primal_conical_hom:np.ndarray)->bool:
    """
    Raises an assertion error if the primal conical is infeasible
    Assumed is the form:
    | A     B/2     D   |
    | B/2   C       E/2 |
    | D     E/2     F   |
    Where points are on the ellipse if:
    Ax + Bxy + Cx + Dx + Ey + F = 0

    :param primal_conical_hom: The matrix to be checked
    :return: True
    """
    det_a_33 = np.linalg.det(primal_conical_hom[:2, :2])
    det_a_q = np.linalg.det(primal_conical_hom)

    assert det_a_q == 0, f"Ellipse is degenerate, det(A_Q) must not be 0, is: {det_a_q}"
    assert det_a_33 > 0, f"Primal conical is not an ellipse: 0 < det(A_33) = {det_a_33}"
    assert det_a_33*det_a_q < 0, f"Ellipse is not real"
    assert np.allclose(primal_conical_hom, primal_conical_hom.T), "Matrix must be symmetric"
    return True

def assert_primal_quadratic_hom_ellipsoid(primal_quadratic_hom:np.ndarray)->bool:
    """
    Raises an assertion error if the primal ellipsoid is infeasible
    Assumed is the form:
    | A     D/2     E/2     G/2|
    | D/2   B       F/2     H/2|
    | E/2   F/2     C       I/2|
    | G/2   H/2     I/2     J  |

    Where points are on the ellipsoid if:
    Axx + Byy + Czz + Dxy + Exz + Fyz + Gx + Hy + Iz + J= 0

    :param primal_quadratic_hom: The matrix to be checked
    return True
    """
    assert primal_quadratic_hom.shape == (4,4), f"Must be 4x4: {primal_quadratic_hom.shape}"
    assert np.allclose(primal_quadratic_hom, primal_quadratic_hom.T), "Matrix must be symmetric"

    return True

def assert_gaussian_ellipse(mu:np.ndarray, sigma:np.ndarray)->bool:
    """
    An ellipsoid that is represented by a Gaussian with mean mu and standard deviation sigma.
    The boundary of the ellipsoid is where p(x) = se
    :param mu: mean of the distribution & ellipsoid
    :param sigma: standard deviation of the distribution & ellipsoid
    """
    n = mu.shape[0]
    assert mu.shape == (n,), f"mu must be vector, is:{mu.shape}"
    assert sigma.shape == (n,n), f"sigma must be {n}x{n}, is:{sigma.shape}"

    assert np.allclose(sigma, sigma.T), f"Cov matrix must be symmetric: \n {sigma}"
    assert np.all(np.diag(sigma) >= 0), f"Variances must be >= 0: {sigma}"

    return True

def fit_ellipsoid_to_3d_point_cloud(
        point_cloud:np.ndarray,
        visualize:bool = True
    )->tuple[np.ndarray, np.ndarray]:
    """
    :param point_cloud: A Nx3 point cloud in the base_frame
    :param visualize: If the fitting should be 3d visualized
    :return: a tuple of the base_t_ellipsoid hom. mtx (4x4) and the primal quadratics (4x4)
    """
    assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 3

    point_cloud = remove_outliers_from_point_cloud(point_cloud, contamination=0.1)

    # Get center
    center = point_cloud.mean(axis=0)  # (3,)
    pts_centered = point_cloud - center

    cov_matrix = np.cov(pts_centered, rowvar=False)

    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    idx = np.argsort(eigvals)[::-1]

    if np.linalg.det(eigvecs[:, idx]) < 0:
        eigvecs[:, -1] *= -1

    base_t_ellipsoid = r_t_to_hom(eigvecs[:, idx], center)

    # to local orientation
    pts_local = (base_t_ellipsoid[:3, :3] @ pts_centered.T).T  # (N,3)

    # a b theta in local coordinate
    a = np.max(np.abs(pts_local[:, 0]))
    b = np.max(np.abs(pts_local[:, 1]))
    c = np.max(np.abs(pts_local[:, 2]))

    # quadric in world coordinate
    q_local = np.diag([1/a**2, 1/b**2, 1/c**2, -1.0])

    ellipsoid_t_base = np.linalg.inv(base_t_ellipsoid)
    primal_quadratic = ellipsoid_t_base.T @ q_local @ ellipsoid_t_base

    if visualize:
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
        to_vis = [base_frame]
        pc_np = sample_points_in_primal_quadratic(base_t_ellipsoid, primal_quadratic)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)
        pcd.paint_uniform_color([0,1,0])

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(point_cloud)
        pcd1.paint_uniform_color([1,0,0])

        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
        frame.transform(base_t_ellipsoid)

        to_vis += [pcd, pcd1, frame]
        o3d.visualization.draw_geometries(to_vis, f"Ellipsoid fit visualization")    

    return base_t_ellipsoid, primal_quadratic

def sample_points_in_primal_quadratic(
        base_t_ellipsoid:np.ndarray, 
        primal_quadratic:np.ndarray,
        resolution:int = 20
    ):
    """
    Generates points on the surface of a primal quadratic, needs base_t_ellipsoid to be more efficient.
    :param base_t_ellipsoid: A 4x4 homogeneous transformation matrix
    :param primal_quadratic: The 4x4 primal quadratic matrix
    :param resolution: The resolution of the point cloud along both rotational axis
    :return: a point cloud consisting of resolution^2 points
    """
    assert_homogeneous_mat(base_t_ellipsoid, size = 4)
    assert_primal_quadratic_hom_ellipsoid(primal_quadratic)

    # [1/a**2, 1/b**2, 1/c**2, -1.0]
    ellipsoid_t_base = np.linalg.inv(base_t_ellipsoid)
    abc1 = np.linalg.inv(ellipsoid_t_base.T) @ primal_quadratic @ np.linalg.inv(ellipsoid_t_base)

    a = np.sqrt(1/abc1[0,0])
    b = np.sqrt(1/abc1[1,1])
    c = np.sqrt(1/abc1[2,2])

    theta = np.linspace(0, np.pi, resolution)
    psi = np.linspace(0, 2*np.pi, resolution)

    theta_grid, psi_grid = np.meshgrid(theta, psi)
    
    points = np.column_stack(
        [
            (a * np.sin(theta_grid) * np.cos(psi_grid)).ravel(),
            (b * np.sin(theta_grid) * np.sin(psi_grid)).ravel(),
            (c * np.cos(theta_grid)).ravel(),
            np.ones(resolution*resolution)
        ]
    )
    base_points_hom = (base_t_ellipsoid @ points.T).T
    return base_points_hom[:, :3]
 

def visualize_primal_quadratics(
        base_t_ellipsoids:np.ndarray, 
        primal_quadratics:np.ndarray,
        bg_point_cloud:np.ndarray | None = None,
        bg_point_cloud_colors:np.ndarray | None = None
    ):
    """
    :param base_t_ellipsoids: Nx4x4 homogeneous transformation matrix from the base to the ellipsoid frames
    :param primal_quadratics: Nx4x4 primal quadratics of the ellipsoids
    :param bg_point_cloud: Mx3-float point cloud to display
    :param bg_point_cloud_colors: Mx3-uint8 RGB color cloud to display
    """
    assert base_t_ellipsoids.shape[0] == primal_quadratics.shape[0]
    assert all(assert_homogeneous_mat(m, size = 4) for m in base_t_ellipsoids)
    assert all(assert_primal_quadratic_hom_ellipsoid(m) for m in primal_quadratics)

    assert bg_point_cloud is None or bg_point_cloud.ndim == 2 and bg_point_cloud.shape[-1] == 3
    assert bg_point_cloud_colors is None or bg_point_cloud.shape == bg_point_cloud_colors.shape

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

    to_vis = [base_frame]

    for b_t_e, primal_quad in zip(base_t_ellipsoids, primal_quadratics):
        pc_np = sample_points_in_primal_quadratic(b_t_e, primal_quad)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)
        to_vis.append(pcd)
    
    if bg_point_cloud is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(bg_point_cloud)

        if bg_point_cloud_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(bg_point_cloud_colors.astype(float)/255.0)
        to_vis.append(pcd)

    o3d.visualization.draw_geometries(to_vis, f"3D features visualization")    
    return 

def fuse_ellipsoids(base_t_ellipsoid_s:np.ndarray, primal_quadratic_s:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    Fuses multiple ellipsoids into one, by generating a point cloud on its shells and fitting an ellipsoid to this point_cloud.
    :param base_t_ellipsoid_s: Nx4x4 homogeneous transformation matrix from the base to the ellipsoid frames
    :param primal_quadratic_s: Nx4x4 primal quadratic of the ellipsoids
    :return: An 4x4 base_t_ellipsoid and 4x4 primal_quadratic
    """
    assert base_t_ellipsoid_s.shape[0] == primal_quadratic_s.shape[0]
    assert all(assert_homogeneous_mat(m, size = 4) for m in base_t_ellipsoid_s)
    assert all(assert_primal_quadratic_hom_ellipsoid(m) for m in primal_quadratic_s)

    point_cloud = np.concatenate([
        sample_points_in_primal_quadratic(b_t_e, p_q, resolution=20) 
        for b_t_e, p_q in zip(base_t_ellipsoid_s, primal_quadratic_s)
    ], axis = 0)

    return fit_ellipsoid_to_3d_point_cloud(point_cloud= point_cloud, visualize=False)


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

    return base_t_ellipsoid_s, primal_quaddratics

def fit_ellipsoid_to_2d_point_cloud(point_cloud:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    Creates a primal conic and base_t_ellipsoid matrix from a 2d point cloud.
    :param point_cloud: A Nx2 point cloud
    :return a tuple of the base_t_ellipsoid hom. mtx (3x3) and the primal conic (3x3)
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
    :param primal_quadratic: The primal quadratic as a 4x4 matrix
    :param cam_t_base: A 4x4 camera->base homogeneous transformation matrix
    :param intrinsic_mtx: The 3x3 intrinsic matrix of the camera

    :return: The 3x3 primal conic of the projected ellipse
    """
    assert_homogeneous_mat(cam_t_base, size=4)
    assert_intrinsic_mat(intrinsic_mtx)

    dual_quadratic = np.linalg.inv(primal_quadratic)
    cam_dual_conic = (intrinsic_mtx @ cam_t_base[:3,:]) @ dual_quadratic @ (intrinsic_mtx @ cam_t_base[:3,:]).T
    cam_primal_conic = np.linalg.inv(cam_dual_conic)

    return cam_primal_conic


def primal_conic_to_gaussian_ellipse(primal_conic:np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    """
    Takes a 3x3 primal conic and returns a normal distribution where p(x) = 0.95, is the ellipsoid
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
        colors:list[str]|str = 'green',
        line_widths:list[int]|int = 1
):
    """
    Takes an 2d gaussian ellipse and turns it into a matplotlib patch ellipse
    :param gaussian_ellipse_s: A list of [mu, sigma] tuples
    :param colors: either one color for all, or a list of ones for each
    :param line_widhts: either one for all, or a list of ones for each
    :return list of plottable ellipses
    """
    assert isinstance(gaussian_ellipse_s, list)
    assert all([isinstance(t,tuple) for t in gaussian_ellipse_s])
    assert all([mu.shape == (2,0) and sigma.shape == (2,2) for mu, sigma in gaussian_ellipse_s])

    assert isinstance(colors, str) or (isinstance(colors, list) and len(colors) == len(gaussian_ellipse_s))
    assert isinstance(line_widths, int) or (isinstance(line_widths, list) and len(line_widths) == len(gaussian_ellipse_s))


    ellipses = []
    for i, (mu, sigma) in enumerate(gaussian_ellipse_s):
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
            edgecolor=colors[i] if isinstance(gaussian_ellipse_s, list) else colors,
            facecolor='none',
            linewidth=line_widths[i] if isinstance(gaussian_ellipse_s, list) else line_widths
        )
        ellipses.append(ellipse)
    return ellipses

def plot_ellipses(
        bgr_img:np.ndarray,
        ellipses:list[Ellipse]
):
    """
    Plots the ellipses onto the background image
    """
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
            debug_vis_3d=True
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



        primal_conics = [
            project_3d_ellipsoid_to_img_coordinates(
                    primal_quadratic=pq,
                    cam_t_base=cam2_t_base_pnp,
                    intrinsic_mtx=self.cam2_intrinsic_mtx,
                ) 
            for pq in self.quadratic_ellipsoid_s
        ]

        gaussian_ellipses = [primal_conic_to_gaussian_ellipse(pc) for pc in primal_conics]
        matplotlib_ellipses = [gaussian_ellipse_s_to_matplotlib_ellipse_s(ge) for ge in gaussian_ellipses]

        plot_ellipses(
            bgr_img=cam2_bgr_image,
            ellipses=matplotlib_ellipses
        )
    
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