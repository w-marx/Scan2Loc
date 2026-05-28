import numpy as np
import os, sys
import open3d as o3d
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utilities import *
import cv2
import scipy

##########################################
## Assertions ############################
##########################################

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

    assert det_a_q != 0, f"Ellipse is degenerate, det(A_Q) must not be 0, is: {det_a_q}"
    assert det_a_33 > 0, f"Primal conical is not an ellipse: 0 < det(A_33) = {det_a_33}"
    assert det_a_33*det_a_q < 0, f"Ellipse is not real: \n {primal_conical_hom}"
    assert np.allclose(primal_conical_hom, primal_conical_hom.T), f"Matrix must be symmetric \n: {primal_conical_hom}"
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

def assert_gaussian_ellipse(mu:np.ndarray, sigma:np.ndarray, dim:int|None = None)->bool:
    """
    An ellipsoid that is represented by a Gaussian with mean mu and standard deviation sigma.
    The boundary of the ellipsoid is where p(x) = se
    :param mu: mean of the distribution & ellipsoid
    :param sigma: standard deviation of the distribution & ellipsoid
    :param dim: If not none this dim will be enforced
    """
    n = mu.shape[0]
    assert dim is None or n == dim, f"mu and sigma sizes: {mu.shape}, {size.shape} dont match {dim}D"
    assert mu.shape == (n,), f"mu must be vector, is:{mu.shape}"
    assert sigma.shape == (n,n), f"sigma must be {n}x{n}, is:{sigma.shape}"

    assert np.allclose(sigma, sigma.T, 0.01), f"Cov matrix must be symmetric: \n {sigma}"
    assert np.all(np.diag(sigma) >= 0), f"Variances must be >= 0: {sigma}"

    return True

def assert_gaussian_ellipse_mat(sigma_mu:np.ndarray, dim:int|None = None)->bool:
    n = sigma_mu.shape[0]
    assert sigma_mu.shape == (n, n+1), f"Wrong shape: {sigma_mu.shape}, should be: {n}x{n+1}"
    mu, sigma = gauss_ellipse_mat_to_tuple(sigma_mu)
    assert_gaussian_ellipse(mu = mu, sigma=sigma, dim = dim)
    return True

def gauss_ellipse_mat_to_tuple(sigma_mu:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    Turns a matrix of the shape Nx(N+1) with the structure:
    [sigma | mu]
    into a mu, sigma tuple
    :return: mu as an array of length N and sigma as an array of size (NxN)
    """
    n = sigma_mu.shape[0]
    mu = sigma_mu[:, n]
    sigma = sigma_mu[:, :n]
    return (mu, sigma)

def gauss_ellipse_tuple_to_mat(mu:np.ndarray, sigma:np.ndarray):
    return np.hstack((sigma, mu.reshape(-1, 1)))



##########################################
## Conversions ###########################
##########################################


def project_primal_quadratic_to_primal_conical(
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
    R_cw = cam_t_base[:3, :3]
    t_cw = cam_t_base[:3, 3]

    P = np.zeros((3, 4), dtype=np.float64)
    P[:, :3] = R_cw
    P[:, 3] = t_cw.flatten()


    cam_dual_conic = (intrinsic_mtx @ P) @ dual_quadratic @ (intrinsic_mtx @ P).T
    cam_primal_conic = np.linalg.inv(cam_dual_conic)

    return cam_primal_conic



def primal_conic_to_gaussian_ellipse(primal_conic:np.ndarray)-> np.ndarray:
    """
    Takes a 3x3 primal conic and returns a normal distribution where p(x) = 0.95, is the ellipsoid
    :param primal_conic: The 3x3 primal conic
    :return a 2x3 [sigma | mu] matrix
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

    return gauss_ellipse_tuple_to_mat(mu=mu, sigma=sigma)



def gaussian_ellipse_s_to_matplotlib_ellipse_s(
        gaussian_ellipse_s:np.ndarray,
        colors:np.ndarray|str = 'green',
        line_widths:list[int]|int = 1
):
    """
    Takes an 2d gaussian ellipse and turns it into a matplotlib patch ellipse
    :param gaussian_ellipse_s: A list of [sigma|mu] matrices
    :param colors: either one color for all, or a numpy array of ones for each
    :param line_widhts: either one for all, or a list of ones for each
    :return list of plottable ellipses
    """
    n = gaussian_ellipse_s.shape[0]
    assert all([assert_gaussian_ellipse_mat(sigma_mu, dim = 2) for sigma_mu in gaussian_ellipse_s])

    assert isinstance(colors, str) or colors.shape[0] == n, f"length: {colors.shape[0]} != {n}"
    assert isinstance(line_widths, int) or (isinstance(line_widths, list) and len(line_widths) == n), f"length: {len(colors)} != {n}"

    ellipses = []
    for i, sigma_mu in enumerate(gaussian_ellipse_s):
        mu, sigma = gauss_ellipse_mat_to_tuple(sigma_mu)
        vals, vecs = np.linalg.eigh(sigma)

        order = np.argsort(vals)[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        # Create ellipse patch
        ellipse = Ellipse(
            xy=(mu[0], mu[1]),
            width = 2 * np.sqrt(vals[0]),
            height = 2 * np.sqrt(vals[1]),
            angle= np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0])),
            edgecolor= colors if isinstance(colors, str) else colors[i, :],
            facecolor='none',
            linewidth=line_widths[i] if isinstance(line_widths, list) else line_widths
        )
        ellipses.append(ellipse)
    return ellipses



##########################################
## Fitting ###############################
##########################################


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
    abc1 = np.linalg.inv(np.linalg.inv(base_t_ellipsoid).T) @ primal_quadratic @ base_t_ellipsoid

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



def fit_primal_conic_to_2d_point_cloud(point_cloud:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    Creates a primal conic and base_t_ellipsoid matrix from a 2d point cloud.
    :param point_cloud: A Nx2 point cloud
    :return a tuple of the base_t_ellipsoid hom. mtx (3x3) and the primal conic (3x3)
    """
    assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 2, f"shape: {point_cloud.shape}"

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
    pts_local = (base_t_ellipsoid[:3, :3].T @ pts_centered.T).T  # (N,3)

    # a b theta in local coordinate
    a = np.max(np.abs(pts_local[:, 0]))
    b = np.max(np.abs(pts_local[:, 1]))
    c = np.max(np.abs(pts_local[:, 2]))

    # quadric in world coordinate
    q_ellipsoid = np.diag([1/a**2, 1/b**2, 1/c**2, -1.0])

    ellipsoid_t_base = np.linalg.inv(base_t_ellipsoid)
    primal_quadratic = ellipsoid_t_base.T @ q_ellipsoid @ ellipsoid_t_base

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



##########################################
## Visualisation #########################
##########################################


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







def wasserstein_distance_sq(sigma_mu1, sigma_mu2):
    assert_gaussian_ellipse_mat(sigma_mu1)
    assert_gaussian_ellipse_mat(sigma_mu2, dim=sigma_mu1.shape[0])
    mu1, sigma1 = gauss_ellipse_mat_to_tuple(sigma_mu1)
    mu2, sigma2 = gauss_ellipse_mat_to_tuple(sigma_mu2)

    mean_dist_sq = np.sum((mu1-mu2)**2)
    sigma2_sqrt = scipy.linalg.sqrtm(sigma2)
    s = scipy.linalg.sqrtm(sigma2_sqrt @ sigma1 @ sigma2_sqrt)

    dist_sq = mean_dist_sq + np.linalg.trace(sigma1+sigma2-2*s)
    return dist_sq