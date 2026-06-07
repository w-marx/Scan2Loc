import cv2
import numpy as np

from shared.assertion_helpers import assert_intrinsic_mat, assert_homogeneous_mat, assert_mxnx3_np_uint8_image_batch


def build_intrinsic_mat(fx:float, fy:float, cx:float, cy:float):
    """
    Builds an intrinsic matrix from the given parameters of the style (skew is assumed as 0):

    | fx     0       cx |
    | 0      fy      cy |
    | 0      0       1  |

    :param fx: The focal length along the x-axis
    :param fy: The focal length along the y-axis
    :param cx: The x-coordinate of the principal point
    :param cy: The y-coordinate of the principal point
    """
    assert fx > 0 and fy > 0 and cx > 0 and cy > 0, f"Cam params must be positive: fx, fy, cx, cy:{fx},{fy},{cx},{cy}"
    return np.array(
        [
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ]
    )


def extract_params_from_intrinsic_mat(intrinsic_mat:np.ndarray)->tuple[float, float, float, float]:
    """
    :param intrinsic_mat: A 3x3 intrinsic matrix
    :return: a tuple consisting of: fx, fy, cx, cy
    """
    assert assert_intrinsic_mat(intrinsic_mat)
    return intrinsic_mat[0,0], intrinsic_mat[1,1], intrinsic_mat[0,2], intrinsic_mat[1,2]


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

    assert assert_homogeneous_mat(base_t_camera, size=4)
    assert assert_intrinsic_mat(intrinsics, hxw_img)
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


def crop_images(images:np.ndarray, intrinsic_matrix:np.ndarray, crop_amount:tuple[int, int])->tuple[np.ndarray, np.ndarray]:
    """
    Takes an image + the intrinsic matrix, crops them and returns the new ones (no distortion & pinhole cam assumed)
    :param images: BxHxWx3-uint8 numpy image array, with B > 0
    :param intrinsic_matrix: The 3x3 intrinsic matrix with which the images were taken
    :param crop_amount: A tuple (crop_w, crop_h) of how much to crop from both sides along each direction
    :return: Bx(H-2*crop_h)x(W-2*crop_w)x3-uint8 image array & the new 3x3 intrinsic matrix
    """
    assert assert_mxnx3_np_uint8_image_batch(images)
    assert images.shape[0] > 0
    assert assert_intrinsic_mat(intrinsic_matrix, hxw_img=images.shape[0])
    h, w = images.shape[1:3]
    crop_w, crop_h = crop_amount
    assert crop_w >= 0 and crop_h >= 0, f"Cant crop negative amount: w: {crop_w}, h: {crop_h}"
    assert crop_w < int(w/2)-2 and crop_h < int(h/2)-2, f"cant crop {h}x{w} to {h-2*crop_h}x{w-2*crop_w}, not enough left"

    cropped_images = images[:, crop_h:h-crop_h, crop_w:w-crop_w, :].copy()

    fx, fy, cx, cy = extract_params_from_intrinsic_mat(intrinsic_matrix)

    return cropped_images, build_intrinsic_mat(fx, fy, cx-crop_w, cy - crop_h)


def scale_images(images:np.ndarray, intrinsic_matrix:np.ndarray, new_resolution:tuple[int, int])->tuple[np.ndarray, np.ndarray]:
    """
    Takes an image + the intrinsic matrix, rescaled them and returns the new ones (no distortion & pinhole cam assumed)
    :param images: BxHxWx3-uint8 numpy image array, with B > 0
    :param intrinsic_matrix: The 3x3 intrinsic matrix with which the images were taken
    :param new_resolution: A tuple (new_w, new_h) of of the new resolution
    :return: Bx(new_h)x(new_w)x3-uint8 image array & the new 3x3 intrinsic matrix
    """
    assert assert_mxnx3_np_uint8_image_batch(images)
    assert images.shape[0] > 0
    assert assert_intrinsic_mat(intrinsic_matrix, images[0])
    old_h, old_w = images.shape[1:3]
    new_w, new_h = new_resolution
    w_scale, h_scale = new_w/old_w, new_h/old_h

    assert new_w > 0 and new_h > 0, f"Cant resize to non-positive size: wxh: {new_w}x{new_h}"

    fx, fy, cx, cy = extract_params_from_intrinsic_mat(intrinsic_matrix)

    resized_images = np.stack([
        cv2.resize(img, (new_w, new_h), interpolation = cv2.INTER_AREA if (w_scale < 1 or h_scale < 1) else cv2.INTER_LINEAR)
        for img in images
    ], axis = 0)

    return resized_images, build_intrinsic_mat(fx*w_scale, fy*h_scale, cx*w_scale, cy*h_scale)
