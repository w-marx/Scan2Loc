import os

import cv2 as cv
import numpy as np
from tqdm import tqdm

DICTIONARY = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
DICTIONARY_ID = 33

def generate_marker_image(outpath: str = "marker.png"):
    """
    Saves an image of the currently used marker to the outpath
    :param outpath: a string of the folder+image name where the .png image will be saved
    """
    marker_image = cv.aruco.generateImageMarker(DICTIONARY, DICTIONARY_ID, 200)
    cv.imwrite(outpath, marker_image)


def get_marker_mask(image:np.ndarray) -> np.ndarray:
    """
    :param image: WxHx3-uint8 RGB image as an numpy array
    :return: WxH-bool mask which is True where the marker is
    """
    detector_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(DICTIONARY, detector_params)
    marker_corners, marker_ids, reject_candidates = detector.detectMarkers(image)
    mask = np.full(image.shape[:2], fill_value=True,dtype="bool")

    for polygon in marker_corners:
        for x in range(len(mask)):
            for y in range(len(mask[x])):
                mask[x][y] = mask[x][y] and cv.pointPolygonTest(polygon, (y,x), False) <= 0
    return mask

def calculate_camera_params_from_images(images: np.ndarray, pattern_size:tuple[int,int] = (9,6), square_size:float = 0.03, visualize_corners:bool = False) -> tuple[np.ndarray, np.ndarray]:
    """
    :param images: NxWxHx3-uint8 RGB images
    :param pattern_size: tuple of number of interior corners on the chessboard
    :param square_size: size of a square on the chessboard in meters
    :param visualize_corners: boolean flag whether to visualize the corners of the chessboard
    :return: 3x3 camera matrix and a vector of distortion coefficients
    """
    objp = np.zeros((pattern_size[0]*pattern_size[1],3), np.float32)
    objp[:, :2] = np.mgrid[0:pattern_size[0],0:pattern_size[1]].T.reshape(-1,2) * square_size

    obj_points = []
    img_points = []

    for image in images:
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        # Directly computes with subpixel accuracy
        ret, corners = cv.findChessboardCornersSB(gray, pattern_size, None)
        if ret:
            obj_points.append(objp)
            img_points.append(corners)
            if visualize_corners:
                cv.drawChessboardCorners(image, pattern_size, corners, ret)
                cv.imshow('Corners2', image)
                cv.waitKey(0)
                cv.destroyAllWindows()

    if len(img_points) == 0:
        raise Exception("No chessboards for calibration found")

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    return mtx, dist


def undistort_images(images:np.ndarray, mtx:np.ndarray, dist:np.ndarray)-> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param images: A NxWxHx3-uint8 array of RGB images
    :param mtx: the current distortion matrix
    :param dist: the current distortion coefficecients
    :return: The new camera matrix, a zero vector (new distortion coefficients that are now 0) and the undistorted images
    """
    print(f"Old image shape: {np.shape(images)}")
    img_size = images.shape[1:3]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, img_size, 0, img_size)
    xn,yn,wn,hn = roi
    undistorted_images = np.array([cv.undistort(img, mtx, dist, None, new_camera_mtx)[xn:xn+wn,yn:yn+hn] for img in images])
    print(f"Undistorted image shape: {np.shape(undistorted_images)}")
    return new_camera_mtx, np.zeros_like(dist), undistorted_images



def estimate_camera_aruco_pose(images: np.ndarray, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray, marker_side_length: float = 0.1) -> list[np.ndarray]:
    """
    :param images: NxWxHx3-uint8 RGB images
    :param camera_matrix: 3x3 camera matrix
    :param distortion_coefficients: array of the distortion coefficients of the camera
    :param marker_side_length: side length of the aruco marker in meters
    :return: a list of 4x4 pose estimates
    """
    detector_params = cv.aruco.DetectorParameters()
    detector = cv.aruco.ArucoDetector(DICTIONARY, detector_params)

    marker_points = np.array([
        [-marker_side_length / 2, marker_side_length / 2, 0],
        [marker_side_length / 2, marker_side_length / 2, 0],
        [marker_side_length / 2, -marker_side_length / 2, 0],
        [-marker_side_length / 2, -marker_side_length / 2, 0],
    ])

    pose_estimates = []
    for index, image in  enumerate(images):
        marker_corners, marker_ids, reject_candidates = detector.detectMarkers(image)

        if len(marker_corners) > 1:
            raise Exception("More then one aruco marker detected, single pose is not calculatable")
        if len(marker_corners) == 0:
            cv.imshow("Marker not found", image)
            cv.waitKey(0)
            cv.destroyWindow("Marker not found")
            raise Exception(f"No aruco marker detected on image: {index}, pose is not calculatable")

        _, rvec, tvec = cv.solvePnP(
            objectPoints=marker_points,
            imagePoints=marker_corners[0][0],
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
            flags=cv.SOLVEPNP_ITERATIVE
        )

        transformation = np.eye(4)
        transformation[:3, :3] = cv.Rodrigues(rvec)[0]
        transformation[:3, 3] = tvec.flatten()
        pose_estimates.append(transformation)
    return pose_estimates






if __name__ == "__main__":
    # marker removal showcase
    glasses_image = cv.imread("in_data/headset/00.png")
    mask = get_marker_mask(glasses_image).astype(np.uint8)
    int_mask = np.stack([mask, mask, mask], axis=2)
    cv.imwrite("image_masked.png", glasses_image * int_mask)

    # calibration
    image_folder = "./in_data/robot_calibration"
    image_names = [f"{image_folder}/{filename}" for filename in os.listdir(image_folder) if filename.endswith('.png')]
    images = np.array([cv.cvtColor(cv.imread(name), cv.COLOR_BGR2RGB) for name in image_names])
    print(f"images: {np.shape(images)}")
    mtx, dist_coef = calculate_camera_params_from_images(images)

    # pose estimation
    image_folder = "./in_data/robot"
    image_names = [f"{image_folder}/{filename}" for filename in os.listdir(image_folder) if filename.endswith('.png')]
    images = np.array([cv.cvtColor(cv.imread(name), cv.COLOR_BGR2RGB) for name in image_names])
    estimate_camera_aruco_pose(images, mtx, dist_coef, 0.1)

