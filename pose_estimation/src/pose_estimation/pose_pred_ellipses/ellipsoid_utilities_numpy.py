import open3d as o3d
from matplotlib.patches import Ellipse
from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import cv2
import scipy
from scipy.optimize import linear_sum_assignment
import numpy as np
from dataclasses import dataclass
import logging

from shared.se3_utilities import r_t_to_hom
from shared.assertion_helpers import assert_homogeneous_mat, assert_intrinsic_mat

##########################################
## Assertions ############################
##########################################

def assert_primal_conical_hom_ellipse(primal_conical_hom:np.ndarray, atol:float = 0.01)->bool:
    """
    Raises an assertion error if the primal conical is infeasible
    Assumed is the form:
    | A     B/2     D/2 |
    | B/2   C       E/2 |
    | D/2   E/2     F   |
    Where points are on the ellipse if:
    Axx + Bxy + Cyy + Dx + Ey + F = 0

    :param primal_conical_hom: The matrix to be checked
    :param atol: The absolute tolerance for all chest
    :return: True
    """
    det_a_33 = np.linalg.det(primal_conical_hom[:2, :2])
    det_a_q = np.linalg.det(primal_conical_hom)

    assert det_a_q != 0, f"Ellipse is degenerate, det(A_Q) must not be 0, is: {det_a_q}"
    assert det_a_33 > -atol, f"Primal conical is not an ellipse: 0 < det(A_33) = {det_a_33}"
    assert det_a_33 * det_a_q < atol, f"Ellipse is not real: \n {primal_conical_hom}"
    assert np.allclose(primal_conical_hom, primal_conical_hom.T, atol=atol), f"Matrix must be symmetric \n: {primal_conical_hom}"
    return True

def assert_primal_conical_hom_ellipse_batch(primal_conical_hom_batch:np.ndarray, atol:float = 0.01)->bool:
    """
    Assert that the given batch contains N, primal conical ellipses
    :param primal_conical_hom_batch: The batch to be checked
    :param atol: The absolute tolerance for all chest
    :return: True
    """
    assert primal_conical_hom_batch.ndim == 3
    return all([assert_primal_conical_hom_ellipse(m, atol=atol) for m in primal_conical_hom_batch])

def assert_primal_quadratic_hom_ellipsoid(primal_quadratic_hom:np.ndarray, atol:float = 0.01)->bool:
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
    :param atol: The absolute tolerance for all tests
    :return: True
    """
    assert primal_quadratic_hom.shape == (4,4), f"Must be 4x4: {primal_quadratic_hom.shape}"
    assert np.allclose(primal_quadratic_hom, primal_quadratic_hom.T, atol=atol), f"Matrix must be symmetric: \n {primal_quadratic_hom}"
    return True

def assert_primal_quadratic_hom_ellipsoid_batch(primal_quadratic_hom_batch:np.ndarray, atol:float = 0.01)->bool:
    """
    Checks if primal_quadratic_hom_batch is a primal quadratic ellipsoid batch
    :param primal_quadratic_hom_batch: The batch to be checked
    :param atol: The absolute tolerance for all tests
    :return: True
    """
    assert primal_quadratic_hom_batch.ndim == 3, f"Must be 3 dimensional: {primal_quadratic_hom_batch.shape}"
    return all([assert_primal_quadratic_hom_ellipsoid(m, atol=atol) for m in primal_quadratic_hom_batch])

def assert_gaussian_ellipse(mu:np.ndarray, sigma:np.ndarray, dim:int|None = None, r_tol:float = 0.01)->bool:
    """
    An ellipsoid that is represented by a Gaussian with mean mu and standard deviation sigma.
    The boundary of the ellipsoid is at a Mahalanobis distance of 1
    :param mu: mean of the distribution & ellipsoid
    :param sigma: standard deviation of the distribution & ellipsoid
    :param dim: If not none this dim will be enforced
    :param r_tol: The relative tolerance for all tests
    :return: True
    """
    n = mu.shape[0]
    assert dim is None or n == dim, f"mu and sigma sizes: {mu.shape}, {sigma.shape} dont match {dim}D"
    assert mu.shape == (n,), f"mu must be vector, is:{mu.shape}"
    assert sigma.shape == (n,n), f"sigma must be {n}x{n}, is:{sigma.shape}"

    assert np.allclose(sigma, sigma.T, r_tol), f"Cov matrix must be symmetric: \n {sigma}"
    assert np.all(np.diag(sigma) >= 0), f"Variances must be >= 0: {sigma}"

    return True

def assert_gaussian_ellipse_mat(sigma_mu:np.ndarray, dim:int|None = None, r_tol:float = 0.01)->bool:
    """
    Asserts that a matrix is of the shape:
    [sigma | mu]
    :param sigma_mu: The matrix to be checked
    :param dim: the dimension of the gaussian, will be enforced if not None
    :param r_tol: The relative tolerance for all tests
    :return: True
    """
    n = sigma_mu.shape[0]
    assert sigma_mu.shape == (n, n+1), f"Wrong shape: {sigma_mu.shape}, should be: {n}x{n+1}"
    mu, sigma = gauss_ellipse_mat_to_tuple(sigma_mu)
    assert assert_gaussian_ellipse(mu = mu, sigma=sigma, dim = dim, r_tol=r_tol)
    return True

def assert_gaussian_ellipse_mat_batch(sigma_mu_s:np.ndarray, dim:int|None = None, r_tol:float = 0.01)->bool:
    """
    Asserts that sigma_mu is a batch of [sigma | mu] Nx(N+1) matrices.
    :param sigma_mu_s: The batch to be checked
    :param dim: the dimension of the gaussian, will be enforced if not None
    :param r_tol: The relative tolerance for all tests
    :return: True
    """
    assert sigma_mu_s.ndim == 3, f"Wrong shape: {sigma_mu_s.shape}, should be: 3D"
    assert all([assert_gaussian_ellipse_mat(sigma_mu, dim = dim, r_tol=r_tol) for sigma_mu in sigma_mu_s])
    return True

def assert_gaussian_ellipse_tuple_batch(mu_s:np.ndarray,sigma_s:np.ndarray, dim:int|None = None, r_tol:float = 0.01)->bool:
    """
    Asserts that sigma_s and mu_s is a batch of sigma, mu Nx(N+1) matrices.
    :param mu_s: The mean batch
    :param sigma_s: The covariance batch
    :param dim: the dimension of the gaussian, will be enforced if not None
    :param r_tol: The relative tolerance for all tests
    :return: True
    """
    assert sigma_s.shape[0] == mu_s.shape[0], f"Sigma: {sigma_s.shape}, Mu: {mu_s.shape} dont match"
    assert sigma_s.ndim == 3 and mu_s.ndim == 2, f"Sigma: {sigma_s.shape}, Mu: {mu_s.shape} not valid"
    assert dim is None or (sigma_s.shape[-2:] == (dim, dim) and mu_s.shape[-1] == dim), f"Sigma: {sigma_s.shape}, Mu: {mu_s.shape} not {dim} dim"
    assert assert_gaussian_ellipse_mat_batch(gauss_ellipse_batch_tuple_to_mat_batch(mu_s=mu_s, sigma_s=sigma_s))
    return True

def assert_valid_prim_quad_2_conic_projection(primal_quadratic:np.ndarray, primal_conic:np.ndarray, atol:float = 1e-8):
    """
    Asserts sanity checks for an projection (e.g. if the primal conic is degenerate)
    :param primal_quadratic: 4x4 primal quadratic ellipsoid
    :param primal_conic: The projected 3x3 primal conic
    """    
    assert assert_primal_quadratic_hom_ellipsoid(primal_quadratic_hom=primal_quadratic)
    assert np.allclose(primal_conic, primal_conic.T, atol=atol), f"Matrix must be symmetric \n: {primal_conic} proj from: \n {primal_quadratic}"    

    det_a_33 = np.linalg.det(primal_conic[:2, :2])
    det_a_q = np.linalg.det(primal_conic)
    eigvals = np.linalg.eigvalsh(primal_conic[:2,:2])
    cond = np.linalg.cond(primal_conic)

    diag_info = (
        f"det(A33)={det_a_33}\n"
        f"det(Q)={det_a_q}\n"
        f"eig(A33)={eigvals}\n"
        f"cond(Q)={cond:.2e}\n"
        f"conic=\n{primal_conic}\n"
        f"quadric=\n{primal_quadratic}"
    )
    
    assert cond < 1e12, f"Ellipse cond to big, numerical limit problems \n {diag_info}"
    assert np.all(eigvals > atol), f"Ellipse quadratic part not positive definite \n {diag_info}"
    assert abs(det_a_q) > atol, f"Ellipse is nearly singular: \n {diag_info}"
    assert det_a_33 > 0, f"Primal conical is not an ellipse: 0 !< det(A_33): \n {diag_info}"
    assert det_a_33 * det_a_q < 0, f"Ellipse is not real: \n {diag_info}"


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
    return mu, sigma

def gauss_ellipse_mat_batch_to_tuple(sigma_mu:np.ndarray)->tuple[np.ndarray, np.ndarray]:
    """
    Turns a batch of matrices of the shape Mx(Nx(N+1)) with the structure:
    [[sigma | mu], ...]
    into a Mxmu, Mxsigma tuple
    :return: mu's as an array of size MxN and sigma's as an array of size Mx(NxN)
    """
    n = sigma_mu.shape[1]
    return sigma_mu[:,:, n], sigma_mu[:, :, :n]

def gauss_ellipse_tuple_to_mat(mu:np.ndarray, sigma:np.ndarray):
    """
    Turns a tuple of sigma (NxN) and mu (N) into a matrix of the shape (NxN+1) with the structure:
    [sigma | mu]
    :param mu: The mean of the ellipsoid
    :param sigma: The sigma of the ellipsoid-distribution
    """
    return np.hstack((sigma, mu.reshape(-1, 1)))

def gauss_ellipse_batch_tuple_to_mat_batch(mu_s: np.ndarray, sigma_s: np.ndarray):
    """
    turns BxN mu vectors and BxNxN sigma matrices into BxNx(N+1)
    :param mu_s: (B, N) batch of means
    :param sigma_s: (B, N, N) batch of covariance matrices
    :return: (B, N, N+1) concatenated matrices
    """
    mu_expanded = mu_s[..., None]
    return np.concatenate([sigma_s, mu_expanded], axis=-1)


##########################################
## Conversions ###########################
##########################################


def normalize_gaussians(sigma_mu_s:np.ndarray, w:int, h:int)->np.ndarray:
    assert assert_gaussian_ellipse_mat_batch(sigma_mu_s, dim=2)

    scaled_sigma_mu_s = sigma_mu_s.copy()
    scaled_sigma_mu_s[:, 0, 2] /= w
    scaled_sigma_mu_s[:, 1, 2] /= h
    scaled_sigma_mu_s[:, 0, 0] /= w*w
    scaled_sigma_mu_s[:, 1, 1] /= h*h
    scaled_sigma_mu_s[:, 1, 0] /= w*h
    scaled_sigma_mu_s[:, 0, 1] /= w*h
    return scaled_sigma_mu_s


def project_primal_quadratics_to_primal_conicals(
        primal_quadratics:np.ndarray,
        cam_t_base:np.ndarray,
        intrinsic_mtx:np.ndarray,
)->np.ndarray:
    """
    Takes a batch of Nx4x4 primal quadratics in the world frame
    and returns the Nx3x3 primal conics in the camera frame
    :param primal_quadratics: The primal quadratics as a Nx4x4 matrix
    :param cam_t_base: A 4x4 camera->base homogeneous transformation matrix
    :param intrinsic_mtx: The 3x3 intrinsic matrix of the camera

    :return: The Nx3x3 primal conic batch of projected ellipse
    """
    assert assert_primal_quadratic_hom_ellipsoid_batch(primal_quadratics)
    assert assert_homogeneous_mat(cam_t_base, size=4)
    assert assert_intrinsic_mat(intrinsic_mtx)

    dual_quadratics = np.linalg.inv(primal_quadratics)

    cam_transform = intrinsic_mtx @ cam_t_base[:3, :]
    cam_dual_conic = np.einsum('ik,nkl,jl->nij', cam_transform, dual_quadratics, cam_transform, optimize=True)
    cam_primal_conic = np.linalg.inv(cam_dual_conic)

    return cam_primal_conic


def filter_good_primal_conicals(primal_conicals:np.ndarray, cond:float = 1e10, min_eigenvalue:float = 1e-8)->np.ndarray:
    """
    Takes a batch of Bx3x3 primal conics and returns those that are numerical good conditioned
    :param primal_conic_s: A batch of Bx3x3 primal conics
    :param cond: The maximal cond of the primal conical
    :param min_eigenvalue: Minimal eigenvalue, small eigenvalues mean near-singular ellipses
    :return a Batch of B'x3x3 primal conics with B' <= B
    """
    assert primal_conicals.ndim == 3 and primal_conicals.shape[1:] == (3,3), f"wrong shape: {primal_conicals.shape}, should be Bx3x3"
    if primal_conicals.shape[0] == 0:
        return np.empty((0,3,3))
    
    finite = np.all(np.isfinite(primal_conicals), axis=(1, 2))
    eigvals = np.linalg.eigvalsh(primal_conicals[:,:2, :2])
    conds = np.linalg.cond(primal_conicals)

    validmask = finite  & (conds < cond) & (eigvals[:, 0] > min_eigenvalue)

    if np.any(~validmask):
        logging.debug(f"removed {np.sum(~validmask)} / {primal_conicals.shape[0]} primal conicals for bad numerical behaviour")
    
    return primal_conicals[validmask]



def primal_conics_to_gaussian_ellipses(primal_conic_s:np.ndarray)-> tuple[np.ndarray, np.ndarray]:
    """
    Takes a batch of Nx3x3 primal conics and returns normal distribution, with Mahalanobis distance = 1, as the ellipsoids border
    :param primal_conic_s: A batch of Nx3x3 primal conics
    :return a tuple of batches of Nx2 mu's and Nx2x2 sigma's
    """
    assert assert_primal_conical_hom_ellipse_batch(primal_conic_s)
    a_s = primal_conic_s[:,0:2, 0:2]
    b_s = primal_conic_s[:,0:2, 2]
    c_s = primal_conic_s[:,2, 2]
    mu_s = -np.linalg.solve(a_s,b_s[:, :, np.newaxis]).squeeze(-1)

    # normalization
    s = np.einsum('ni,nji,nj->n', mu_s, a_s, mu_s, optimize=True) - c_s
    sigma_s = np.linalg.inv((primal_conic_s / s[:, None, None])[:, 0:2, 0:2])
    
    assert assert_gaussian_ellipse_tuple_batch(mu_s=mu_s, sigma_s=sigma_s)

    return mu_s, sigma_s


def gaussian_ellipse_s_to_matplotlib_ellipse_s(
        gaussian_ellipse_s:np.ndarray,
        colors:np.ndarray|str = 'green',
        line_widths:np.ndarray|int = 1,
        line_style:str = "--"
):
    """
    Takes multiple 2d gaussian ellipses and turns tem into a matplotlib patch ellipses
    :param gaussian_ellipse_s: A list of [sigma|mu] matrices
    :param colors: either one color for all, or a numpy array of ones for each
    :param line_widths: either one for all, or a numpy ones for each
    :param line_style: The matplotlib line style for the ellipse
    :return list of plottable ellipses
    """
    n = gaussian_ellipse_s.shape[0]
    assert all([assert_gaussian_ellipse_mat(sigma_mu, dim = 2) for sigma_mu in gaussian_ellipse_s])

    assert isinstance(colors, str) or colors.shape[0] == n, f"length: {colors.shape[0]} != {n}"
    assert isinstance(line_widths, int) or line_widths.shape[0] == n, f"length: {colors.shape[0]} != {n}"

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
            linewidth=line_widths if isinstance(line_widths, int) else line_widths[i],
            linestyle=line_style
        )
        ellipses.append(ellipse)
    return ellipses



##########################################
## Fitting ###############################
##########################################


def sample_points_in_primal_conic(
        base_t_ellipsoid:np.ndarray, 
        primal_conic:np.ndarray,
        resolution:int = 20
    ):
    """
    Generates points on the surface of a primal quadratic, needs base_t_ellipsoid to be more efficient.
    :param base_t_ellipsoid: A 4x4 homogeneous transformation matrix
    :param primal_conic: The 3x3 primal conical matrix
    :param resolution: The resolution of the point cloud along both rotational axis
    :return: a point cloud consisting of resolution^2 points
    """
    raise NotImplementedError()


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
    assert assert_homogeneous_mat(base_t_ellipsoid, size = 4)
    assert assert_primal_quadratic_hom_ellipsoid(primal_quadratic)

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


def fit_primal_conic_to_2d_point_cloud(point_cloud:np.ndarray)->np.ndarray:
    """
    Creates a primal conic and base_t_ellipsoid matrix from a 2d point cloud.
    Is very sensitive to outliers.
    :param point_cloud: A Nx2 point cloud
    :return The fitted (3x3) primal conic
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

    return primal_conic


#def fuse_ellipsoids(base_t_ellipsoid_s:np.ndarray, primal_quadratic_s:np.ndarray, config:EllipsoidFittingConfig)->tuple[np.ndarray, np.ndarray] | None:
#    """
#    Fuses multiple ellipsoids into one, by generating a point cloud on its shells and fitting an ellipsoid to this point_cloud.
#    :param base_t_ellipsoid_s: Nx4x4 homogeneous transformation matrix from the base to the ellipsoid frames
#    :param primal_quadratic_s: Nx4x4 primal quadratic of the ellipsoids
#    :return: An 4x4 base_t_ellipsoid and 4x4 primal_quadratic or None if not successful
#    """
#    assert base_t_ellipsoid_s.shape[0] == primal_quadratic_s.shape[0]
#    assert all(assert_homogeneous_mat(m, size = 4) for m in base_t_ellipsoid_s)
#    assert all(assert_primal_quadratic_hom_ellipsoid(m) for m in primal_quadratic_s)
#
#    point_cloud = np.concatenate([
#        sample_points_in_primal_quadratic(b_t_e, p_q, resolution=20) 
#        for b_t_e, p_q in zip(base_t_ellipsoid_s, primal_quadratic_s)
#    ], axis = 0)
#
#    return fit_ellipsoid_to_3d_point_cloud(point_cloud= point_cloud, config=config)


##########################################
## Visualisation #########################
##########################################


def visualize_primal_quadratics(
        base_t_ellipsoid_s:np.ndarray,
        primal_quadratic_s:np.ndarray,
        bg_point_cloud:np.ndarray | None = None,
        bg_point_cloud_colors:np.ndarray | None = None,
        avg_colors:np.ndarray | None = None
    ):
    """
    :param base_t_ellipsoid_s: Nx4x4 homogeneous transformation matrix from the base to the ellipsoid frames
    :param primal_quadratic_s: Nx4x4 primal quadratics of the ellipsoids
    :param bg_point_cloud: Mx3-float point cloud to display
    :param bg_point_cloud_colors: Mx3-uint8 BGR color cloud to display
    """
    assert base_t_ellipsoid_s.shape[0] == primal_quadratic_s.shape[0]
    assert all(assert_homogeneous_mat(m, size = 4) for m in base_t_ellipsoid_s)
    assert all(assert_primal_quadratic_hom_ellipsoid(m) for m in primal_quadratic_s)

    assert bg_point_cloud is None or bg_point_cloud.ndim == 2 and bg_point_cloud.shape[-1] == 3
    assert bg_point_cloud_colors is None or bg_point_cloud.shape == bg_point_cloud_colors.shape

    assert avg_colors is None or avg_colors.shape == (base_t_ellipsoid_s.shape[0], 3), f"Wrong color shape: {avg_colors.shape}"

    base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

    to_vis = [base_frame]

    for i, (b_t_e, primal_quad) in enumerate(zip(base_t_ellipsoid_s, primal_quadratic_s)):
        pc_np = sample_points_in_primal_quadratic(b_t_e, primal_quad)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc_np)

        if avg_colors is not None:
            pcd.paint_uniform_color(avg_colors[i, [2,1,0]].astype(float)/255)

        to_vis.append(pcd)
    
    if bg_point_cloud is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(bg_point_cloud)

        if bg_point_cloud_colors is not None:
            pcd.colors = o3d.utility.Vector3dVector(bg_point_cloud_colors[:, [2,1,0]].astype(float)/255.0)
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

def pairwise_sq_wasserstein_distance(
        mu1_s:np.ndarray,
        sigma1_s:np.ndarray,
        mu2_s: np.ndarray,
        sigma2_s: np.ndarray,
)->np.ndarray:
    """"
    Computes the pairwise wasserstein distance between all normal distribution combinations
    Doesn't use the fact that wasserstein distance is symmetric
    :param mu1_s: A batch of Mx2 gaussian distribution means
    :param sigma1_s: A batch of Mx2x2 covariance matrices
    :param mu2_s: A batch of Nx2 gaussian distribution means
    :param sigma2_s: A batch of Nx2x2 covariance matrices
    :return A NxM matrix of the squared wasserstein distances
    """
    assert assert_gaussian_ellipse_mat_batch(gauss_ellipse_batch_tuple_to_mat_batch(mu1_s, sigma1_s), dim=2)
    assert assert_gaussian_ellipse_mat_batch(gauss_ellipse_batch_tuple_to_mat_batch(mu2_s, sigma2_s), dim=2)

    n, m = mu1_s.shape[0], mu2_s.shape[0]

    sigma2_sqrt_s = mat_sqrt_2x2_batch(sigma2_s) #Mx2x2
    mean_dist_sq_mat = np.sum((mu1_s[:, None, :]-mu2_s[None, :, :])**2, axis=-1) #NxM

    cross_inner_mat = np.einsum('mkl,nla,mab->nmkb',sigma2_sqrt_s, sigma1_s, sigma2_sqrt_s, optimize=True)
    cross_sqrt = mat_sqrt_2x2_batch(cross_inner_mat.reshape(-1,2,2)).reshape(n,m,2,2)

    sigma1_traces = np.einsum('nii->n', sigma1_s, optimize=True)[:, None]
    sigma2_traces = np.einsum('mii->m', sigma2_s, optimize=True)[None, :]
    cross_inner_mat_traces = np.einsum('nmii->nm',cross_sqrt, optimize=True)

    dist_sq = mean_dist_sq_mat + sigma1_traces + sigma2_traces - 2*cross_inner_mat_traces
    return dist_sq


def plot_matchings(
        ax:Axes, 
        rgb_image:np.ndarray, 
        observed_sigma_mu_s:np.ndarray,
        proj_sigma_mu_s:np.ndarray,
        observed_matched_idx_s:np.ndarray,
        proj_matched_idx_s:np.ndarray
    )->None:
    """
    Plots the ellipsoids and matches
    :param ax: The axis to plot upon
    :param rgb_image: The HxWx3-uint8 background image
    :param observed_sigma_mu_s: A Nx2x3 array of [[Sigma_0 | mu_0], ...] gaussians
    :param proj_sigma_mu_s: A Mx2x3 array of [[Sigma_0 | mu_0], ...] gaussians
    :param observed_matched_idx_s: The indices such that obs_sigmas[obs_idx[i]] was matched to proj_sigmas[proj_idx[i]]
    :param proj_matched_idx_s: Array of the same length as observed_matched_idx_s
    :return: None
    """
    ax.imshow(rgb_image)

    elli1_s = gaussian_ellipse_s_to_matplotlib_ellipse_s(observed_sigma_mu_s, line_style="-", line_widths=2, colors='black')
    elli2_s = gaussian_ellipse_s_to_matplotlib_ellipse_s(proj_sigma_mu_s, line_style="--", line_widths=2, colors='black')

    ax.set_title("Matching visualisation")

    for e1 in elli1_s:
        ax.add_patch(e1)
    for e2 in elli2_s:
        ax.add_patch(e2)

    ax.scatter(observed_sigma_mu_s[:,0,2],observed_sigma_mu_s[:,1,2], s=5, marker = 'o', color = 'black')
    ax.scatter(proj_sigma_mu_s[:,0,2],proj_sigma_mu_s[:,1,2], s=5, marker = 's', color = 'black')

    ax.quiver(
        observed_sigma_mu_s[observed_matched_idx_s,0,2],
        observed_sigma_mu_s[observed_matched_idx_s,1,2],
        proj_sigma_mu_s[proj_matched_idx_s,0,2]-observed_sigma_mu_s[observed_matched_idx_s,0,2], 
        proj_sigma_mu_s[proj_matched_idx_s,1,2]-observed_sigma_mu_s[observed_matched_idx_s,1,2],
        angles='xy', scale_units='xy', scale=1,
        color='lime',
        alpha=0.6,
        width=0.005
    )

    empty_lines = [
        mlines.Line2D([], [], color='black', linestyle='-',  linewidth=2, label='Observed'),
        mlines.Line2D([], [], color='black', linestyle='--', linewidth=2, label='Projected'),
    ]
    ax.legend(handles=empty_lines, loc='upper right')


##########################################
## Wasserstein & Distances ###############
##########################################


def mat_sqrt_2x2_batch(matrices:np.ndarray)->np.ndarray:
    """
    Computes a batch of mat^{1/2}
    :param matrices: The Bx2x2 matrix batch
    :return: A Bx2x2 batch of [mat_i^{1/2}, ... ]
    """
    assert matrices.ndim == 3 and matrices.shape[-2:] == (2,2), f"Not Bx2x2: {matrices.shape}"
    vals, vecs = np.linalg.eigh(matrices)
    sqrt_vals = np.sqrt(np.clip(vals, 1e-9, None))
    return np.einsum('bik,bk,bjk->bij', vecs, sqrt_vals, vecs, optimize=True)


def wasserstein_distance_sq(sigma_mu1, sigma_mu2):
    """
    Computes the wasserstein distances between the two 2D normal distributions
    :param sigma_mu1: A normal distribution in the form [sigma | mu] (3x2)
    :param sigma_mu2: A normal distribution in the form [sigma | mu] (3x2)
    :return: The wasserstein distance between N_1, N_2
    """
    assert assert_gaussian_ellipse_mat(sigma_mu1)
    assert assert_gaussian_ellipse_mat(sigma_mu2, dim=sigma_mu1.shape[0])
    mu1, sigma1 = gauss_ellipse_mat_to_tuple(sigma_mu1)
    mu2, sigma2 = gauss_ellipse_mat_to_tuple(sigma_mu2)

    mean_dist_sq = np.sum((mu1-mu2)**2)
    sigma2_sqrt = scipy.linalg.sqrtm(sigma2)
    s = scipy.linalg.sqrtm(sigma2_sqrt @ sigma1 @ sigma2_sqrt)

    dist_sq = mean_dist_sq + np.linalg.trace(sigma1+sigma2-2*s)
    return dist_sq


@dataclass(frozen=True, kw_only=True)
class GaussianMatchingConfig:
    """
    A dataclass describe hot to match gaussians
    :param dummy_value: The cost of no-match, lower -> less False Positives
    :param k: Filter out matching costs >= median_cost + k*MAD 
    """
    dummy_value:float = 0.05
    k:float = 1000

    def __post__init__(self):
        assert self.dummy_value > 0
        assert self.k > 0


def match_gaussians_hungarian_on_wasserstein(
        proj_sigma_mu_s:np.ndarray, 
        obs_sigma_mu_s:np.ndarray,
        image_size:tuple[int, int],
        config:GaussianMatchingConfig = GaussianMatchingConfig(),
        visualize_matching:None | np.ndarray = None
    )->tuple[np.ndarray, np.ndarray]:
    """
    Uses the hungarian algorithm to match distributions
    :param proj_sigma_mu_s: Nx2x3 array of the form [[sigma_1 | mu_1], ... ]
    :param obs_sigma_mu_s: Mx2x3 array of the form [[sigma_1 | mu_1], ... ]
    :param image_size: A tuple of the image size (h,w) where the gaussians come from (used for normalisation)
    :param dummy_value: The value for the no-match alternative
    :param k: valid matches hav the cost: cost < median-cost + k * mad
    :return: 2 arrays of length n, with the indices for the matching gaussians 
    """
    n, m = proj_sigma_mu_s.shape[0], obs_sigma_mu_s.shape[0]
    assert assert_gaussian_ellipse_mat_batch(proj_sigma_mu_s)
    assert assert_gaussian_ellipse_mat_batch(obs_sigma_mu_s)

    h, w = image_size
    proj_sigma_mu_s_n = normalize_gaussians(proj_sigma_mu_s, h, w)
    obs_sigma_mu_s_n = normalize_gaussians(obs_sigma_mu_s, h, w)

    adjecency_mat = np.full((m+n, m+n), config.dummy_value, dtype = np.float32)
    mu1_s, sigma1_s = gauss_ellipse_mat_batch_to_tuple(proj_sigma_mu_s_n)
    mu2_s, sigma2_s = gauss_ellipse_mat_batch_to_tuple(obs_sigma_mu_s_n)
    adjecency_mat[:n, :m] = pairwise_sq_wasserstein_distance(mu1_s, sigma1_s, mu2_s, sigma2_s)

    row_ind, col_ind = linear_sum_assignment(adjecency_mat)
    valid_ind = (col_ind < m) & (row_ind < n)
    row_ind, col_ind = row_ind[valid_ind], col_ind[valid_ind]
    
    median_cost = np.median(adjecency_mat[row_ind, col_ind])
    mad = np.median(np.abs(adjecency_mat[row_ind, col_ind] - median_cost))

    valid_matches = (adjecency_mat[row_ind, col_ind] < median_cost+ config.k * mad)
    proj_matched_idx_s, obs_matched_idx_s = row_ind[valid_matches], col_ind[valid_matches]

    if visualize_matching is not None:
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        plot_matchings(
            ax = ax, 
            rgb_image=visualize_matching, 
            proj_sigma_mu_s=proj_sigma_mu_s,
            proj_matched_idx_s=proj_matched_idx_s,
            observed_sigma_mu_s=obs_sigma_mu_s,
            observed_matched_idx_s=obs_matched_idx_s
        )
        plt.show()
    return proj_matched_idx_s, obs_matched_idx_s