import cv2 as cv
import numpy as np

DICTIONARY = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_6X6_250)
DICTIONARY_ID = 33

def generate_marker_image(outpath: str = "marker.png"):
    """
    Saves an image of the currently used marker to the outpath
    :param outpath: a string of the folder+image name where the .png image will be saved
    """
    marker_image = cv.aruco.generateImageMarker(DICTIONARY, DICTIONARY_ID, 200)
    cv.imwrite(outpath, marker_image)

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
    :param dist: the current distortion coefficients
    :return: The new camera matrix, a zero vector (new distortion coefficients that are now 0) and the undistorted images
    """
    print(f"Old image shape: {np.shape(images)}")
    img_size = images.shape[1:3]
    new_camera_mtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, img_size, 0, img_size)
    xn,yn,wn,hn = roi
    undistorted_images = np.array([cv.undistort(img, mtx, dist, None, new_camera_mtx)[xn:xn+wn,yn:yn+hn] for img in images])
    print(f"Undistorted image shape: {np.shape(undistorted_images)}")
    return new_camera_mtx, np.zeros_like(dist), undistorted_images