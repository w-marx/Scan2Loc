import numpy as np


def assert_intrinsic_mat(m:np.ndarray, hxw_img: np.ndarray | None = None)->bool:
    """
    Asserts that a matrix is of the style:
        |  fx  0   cx  |
        |  0   fy  cy  |
        |  0   0   1   |

    Note that skew is assumed to be 0
    :param m: the intrinsic matrix
    :param hxw_img: an hxw image to extract the dimensions from if available
    :return True
    """
    assert m.shape == (3, 3), f"M not 3x3 {m.shape}"
    assert m[0, 0] > 0 and m[1, 1] > 0, f"fx and fy must be positive"
    assert np.allclose(m[2, :],[0, 0, 1]), f"bottom row of M incorrect: {m[2:]}"
    assert hxw_img is None or 0 < m[0, 2] < hxw_img.shape[1] and 0 < m[1, 2] < hxw_img.shape[0], f"center in wrong location {m[0, 2]}, {m[1, 2]} in {hxw_img.shape}"
    return True


def assert_homogeneous_mat(m:np.ndarray, size:None|int = None, abs_tolerance:float = 0.001) -> bool:
    """
    Asserts that a matrix is NxN and of the style:
    |  SO(N-1)   t  |
    |  0 ... 0   1  |

    :param m: A homogeneous matrix
    :param size: If not none this size will be asserted
    :param abs_tolerance: The tolerance for det = 1 & SO(N) @ SO(N).T = unity matrix
    :return True
    """
    assert isinstance(m, np.ndarray), f"SE3 matrix must be a numpy array, got {type(m)}: {m}"
    n = m.shape[0]
    assert size is None or n == size, f"Matrix is {m.shape} not {size}x{size}"
    assert m.shape == (n, n), f"M not 4x4 {m.shape}"
    assert np.isclose(m[-1,-1], 1), f"Corner 1 missing: {m}"
    assert np.allclose(m[n-1, :-1],np.zeros(n-1)), f"bottom row of M incorrect: {m[n-1, :]}"
    assert np.allclose(m[:n-1, :n-1] @ m[:n-1, :n-1].T, np.eye(n-1), atol=abs_tolerance), f"Rot part not invertible by transpose: {m[:3, :3] @ m[:3, :3].T}"
    assert np.isclose(np.linalg.det(m[:n-1, :n-1]), 1, atol=abs_tolerance), f"Determinant is not 1: {np.linalg.det(m[:3, :3])}"
    return True


def assert_homogeneous_mat_batch(mtx_s:np.ndarray, size:None|int = None, abs_tolerance:float = 0.001) -> bool:
    """
    Asserts that mtx_s is a Nxsizexsize batch of homogeneous matrices, with N >= 0
    :param mtx_s: A batch of homogeneous matrices
    :param size: If not none this size will be asserted
    :param abs_tolerance: The tolerance for det = 1 & SO(N) @ SO(N).T = unity matrix
    :return True
    """
    assert isinstance(mtx_s, np.ndarray), f"hom mtx. batch must be a numpy array, got {type(mtx_s)}"
    assert mtx_s.ndim == 3, f"Invalid shape for hom. batch: {mtx_s.shape}"
    assert all([assert_homogeneous_mat(m, size=size, abs_tolerance=abs_tolerance) for m in mtx_s])
    return True


def assert_mxnx3_np_uint8_image(img:np.ndarray)->bool:
    """
    Asserts that the img has the dimensions mxnx3 with m,n > 0 and the datatype np.uint8
    :return: True
    """
    assert isinstance(img, np.ndarray), f"img must be a numpy array, got {type(img)}"
    assert img.ndim == 3 and img.shape[2] == 3, f"wrong img shape: {img.shape}"
    assert img.shape[0] > 0 and img.shape[1] > 0, f"img is empty: {img.shape}"
    assert img.dtype == np.uint8, f"wrong img dtype: {img.dtype}"
    return True


def assert_mxnx3_np_uint8_image_batch(imgs:np.ndarray)->bool:
    """
    Asserts that the img has the dimensions Nxmxnx3 with m,n > 0 and the datatype np.uint8
    :param imgs: A Nxmxnx3-uint8 image batch
    :returns: True
    """
    assert isinstance(imgs, np.ndarray), f"img must be a numpy array, got {type(imgs)}"
    assert imgs.ndim == 4, f"Image batch needs to be 4D, is: {imgs.shape}"
    assert all([assert_mxnx3_np_uint8_image(img) for img in imgs])
    return True


def assert_mxnx3_np_float_image(img:np.ndarray)->bool:
    """
    Asserts that the img has the dimensions mxnx3 with m,n > 0 and the datatype np.uint8
    mostly meant for images where pixels are 3d cartesian coordinates
    :return: True
    """
    assert isinstance(img, np.ndarray), f"img must be a numpy array, got {type(img)}"
    assert img.ndim == 3 and img.shape[2] == 3, f"wrong img shape: {img.shape} should be mxnx3"
    assert img.shape[0] > 0 and img.shape[1] > 0, f"img is empty: {img.shape}"
    assert np.issubdtype(img.dtype, np.floating), f"wrong img dtype: {img.dtype} should be np.floating"
    return True


def assert_mxnx3_np_float_image_batch(imgs:np.ndarray)->bool:
    """
    Asserts that the img has the dimensions Bxmxnx3 with m,n > 0 and the datatype floating
    mostly meant for images where pixels are 3d cartesian coordinates
    :param imgs: A Bxmxnx3-float image batch
    :returns: True
    """
    assert isinstance(imgs, np.ndarray), f"img must be a numpy array, got {type(imgs)}"
    assert imgs.ndim == 4, f"float image batch needs to be 4D, is: {imgs.shape}"
    assert all([assert_mxnx3_np_float_image(img) for img in imgs])
    return True


def assert_mxn_np_float_image(img:np.ndarray)->bool:
    """
    Asserts that the img has the dimensions mxnx3 with m,n > 0 and the datatype floating
    :return: True
    """
    assert isinstance(img, np.ndarray), f"img must be a numpy array, got {type(img)}"
    assert img.ndim == 2, f"wrong img shape: {img.shape}"
    assert img.shape[0] > 0 and img.shape[1] > 0, f"img is empty: {img.shape}"
    assert np.issubdtype(img.dtype, np.floating), f"wrong img dtype: {img.dtype}"
    return True


def assert_mxn_np_float_image_batch(imgs:np.ndarray)->bool:
    """
    Asserts that the img has the dimensions Bxmxn with m,n > 0 and the datatype floating
    Mostly meant for depth images
    :param imgs: A Bxmxnxfloat image batch
    :returns: True
    """
    assert isinstance(imgs, np.ndarray), f"img must be a numpy array, got {type(imgs)}"
    assert imgs.ndim == 3, f"float image batch needs to be 3D, is: {imgs.shape}"
    all([assert_mxn_np_float_image(img) for img in imgs])
    return True


def assert_bgr_xyz_image_pair_batch(bgr_images:np.ndarray, xyz_images:np.ndarray)->bool:
    """
    Asserts that both batches has shape BxHxWx3 and that bgr_images is uint8 and xyz_images flot
    """
    assert bgr_images.shape == xyz_images.shape, f"bgr_images and xyz_images must have the same shape: {bgr_images.shape}, {xyz_images.shape}"
    assert assert_mxnx3_np_uint8_image_batch(bgr_images)
    assert assert_mxnx3_np_float_image_batch(xyz_images)
    return True


def get_image_type_hxw(img:np.ndarray) -> str:
    """
    Takes an numpy image array and returns its image type (mostly for debugging)
    :param img: NxHxWx...
    :return: portrait/square/landscape
    """
    assert img.ndim >= 2
    if img.shape[0] > img.shape[1]:
        return "portrait"
    if img.shape[0] == img.shape[1]:
        return "square"
    return "landscape"