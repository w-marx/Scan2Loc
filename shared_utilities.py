import numpy as np

def r_t_to_hom(r:np.ndarray, t:np.ndarray) -> np.ndarray:
    """
    Takes a rotation and translation vector and returns the homogeneous transformation matrix
    :param r: rotation matrix size: NxN
    :param t: translation vector size: N
    :return: N+1xN+1 numpy array
    """
    n = t.shape[0]
    assert r.shape == (n, n), f"R not {n}x{n}: {r.shape}"
    assert t.shape == (n,), f"T not {n}: {t.shape}"

    m = np.eye(n+1)
    m[0:n, 0:n] = r
    m[0:n, n] = t
    return m

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
    n = m.shape[0]
    assert size is None or n == size, f"Matrix is {m.shape} not {size}x{size}"
    assert m.shape == (n, n), f"M not 4x4 {m.shape}"
    assert np.isclose(m[-1,-1], 1), f"Corner 1 missing: {m}"
    assert np.allclose(m[n-1, :-1],np.zeros(n-1)), f"bottom row of M incorrect: {m[n-1, :]}"
    assert np.allclose(m[:n-1, :n-1] @ m[:n-1, :n-1].T, np.eye(n-1), atol=abs_tolerance), f"Rot part not invertible by transpose: {m[:3, :3] @ m[:3, :3].T}"
    assert np.isclose(np.linalg.det(m[:n-1, :n-1]), 1, atol=abs_tolerance), f"Determinant is not 1: {np.linalg.det(m[:3, :3])}"
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

def create_3d_camera(
        base_t_camera:np.ndarray,
        intrinsics: np.ndarray,
        hxw_img: np.ndarray,
        scale:float = 0.1
    ):
    """
    Creates an Lineset representing the camera (using the intrinsics)
    :param base_t_camera: 4x4 hom. matrix of the camera location
    :param intrinsics: 3x3 intrinsics matrix of the camera
    :param hxw_img: HxWx... array of the image
    :param scale: size of the camera in base_t_camera units
    :return o3d.geometry.Lineset
    """
    import open3d as o3d

    _ = assert_homogeneous_mat(base_t_camera, size=4)
    _ =  assert_intrinsic_mat(intrinsics, hxw_img)
    assert scale > 1e-6
    assert hxw_img.ndim >= 2

    fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
    w, h = hxw_img.shape[1], hxw_img.shape[0]

    corners_hom = np.array([
        [-cx/fx, -cy/fy, 1.0, 1.0/scale],
        [(w-cx)/fx, -cy/fy, 1.0, 1.0/scale],
        [(w-cx)/fx, (h-cy)/fy, 1.0, 1.0/scale],
        [-cx/fx, (h-cy)/fy, 1.0, 1.0/scale],
        [0,0,0, 1.0/scale]
    ])*scale
    corners = (base_t_camera @ corners_hom.T)[:3, :]
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(corners.T)
    lines.lines = o3d.utility.Vector2iVector(
        [[0,1], [1,2], [2,3], [3,0], [4,0], [4,1], [4,2], [4,3]]
    )
    return lines

def calc_rotational_difference(hom1:np.ndarray, hom2:np.ndarray)->float:
    """
    Computes the rotational difference between two homogeneous 4x4 matrices
    :param hom1: homogeneous 4x4 matrix
    :param hom2: homogeneous 4x4 matrix
    :return rotational difference as float
    """
    assert_homogeneous_mat(hom1, size=4)
    assert_homogeneous_mat(hom2, size=4)

    return np.arccos(np.clip((np.trace(hom1[:3, :3] @ hom2[:3, :3].T) - 1) / 2, -1.0, 1.0))

def compute_pose_pseudo_median(poses:list[np.ndarray])->np.ndarray | None:
    """
    Takes a numpy array of poses and computes the median pose.
    To compute the median pose the median rotation and the geometric median of the translation are combined.
    Therefore, the returned pose may not be in poses
    :param poses: Nx4x4 numpy array of poses
    :return: median pose, as a 4x4 numpy array
    """
    if len(poses) == 0:
        return None
    assert all([assert_homogeneous_mat(m, size=4) for m in poses])
    
    median_pose = np.eye(4)
    median_pose[:3,3] = min(poses, key = lambda x: sum([np.linalg.norm(x[:3,3]-y[:3,3]) for y in poses]))[:3,3]
    median_pose[:3,:3] = min(poses, key = lambda x: sum([calc_rotational_difference(x,y) for y in poses]))[:3,:3]
    return median_pose