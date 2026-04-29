import numpy as np
import cv2


def assemble_homogeneous_matrix(rvec:np.ndarray, tvec:np.ndarray) -> np.ndarray:
    """
    Takes an rotation and translation vector and returns the homogeneous transformation matrix
    :param rvec: rotation vector
    :param tvec: translation vector
    :return: 4x4 numpy array
    """
    transformation = np.eye(4)
    transformation[:3, :3] = cv2.Rodrigues(rvec)[0]
    transformation[:3, 3] = tvec.flatten()
    return transformation


class ArucoCharucoDetector:
    def get_camera_t_marker(self, images:list[np.ndarray], camera_matrix:np.ndarray, distortion_coefficients:list[float])->list[np.ndarray | None]:
        """
        Returns the pose camera_t_marker or for each image in the list as a list of 4x4 homogeneous matrices
        :param images: list of WxHx3 RGB images
        :param camera_matrix: the 3x3 intrinsic camera matrix
        :param distortion_coefficients: the distortion coefficients of the camera
        :return: list of 4x4 homogeneous matrices
        """
        pass

    def remove_markers(self, images:list[np.ndarray])->list[np.ndarray]:
        """
        Returns the images with the markers digitally removed (pixels set to 0)
        :param images: list of WxHx3 RGB images
        :return: list of WxHx3 RGB images without the aruco markers
        """
        pass

class ArucoDetector(ArucoCharucoDetector):
    def __init__(
            self,
            aruco_marker_side_length:float,
            aruco_marker_dictionary,
    ):
        super().__init__()
        self.aruco_marker_side_length = aruco_marker_side_length
        self.aruco_marker_dictionary = aruco_marker_dictionary

    def get_camera_t_marker(self, images:list[np.ndarray], camera_matrix:np.ndarray, distortion_coefficients:list[float])->list[np.ndarray | None]:
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

        detector = cv2.aruco.ArucoDetector(self.aruco_marker_dictionary, detector_params)

        marker_points = np.array([
            [-self.aruco_marker_side_length / 2, self.aruco_marker_side_length / 2, 0],
            [self.aruco_marker_side_length / 2, self.aruco_marker_side_length / 2, 0],
            [self.aruco_marker_side_length / 2, -self.aruco_marker_side_length / 2, 0],
            [-self.aruco_marker_side_length / 2, -self.aruco_marker_side_length / 2, 0],
        ])

        camera_t_aruco_s = []
        for index, image in enumerate(images):
            marker_corners, marker_ids, reject_candidates = detector.detectMarkers(image)

            if len(marker_corners) > 1:
                raise Exception("More then one aruco marker detected, single pose is not calculatable")
            if len(marker_corners) == 0:
                camera_t_aruco_s.append(None)
                continue

            _, rvec, tvec = cv2.solvePnP(
                objectPoints=marker_points,
                imagePoints=marker_corners[0][0],
                cameraMatrix=camera_matrix,
                distCoeffs=distortion_coefficients,
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            camera_t_aruco_s.append(assemble_homogeneous_matrix(rvec=rvec, tvec=tvec))

        return camera_t_aruco_s


class CharucoDetector(ArucoCharucoDetector):
    def __init__(
            self,
            board_size:tuple[int,int] = (14, 9),
            square_size:float = 0.0188,
            marker_size:float = 0.0146,
            aruco_dictionary= cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250),
            min_fraction_of_markers:float = 1.0
    ):
        super().__init__()
        self.board = cv2.aruco.CharucoBoard(board_size, square_size, marker_size, aruco_dictionary)
        self.min_number_of_markers = min_fraction_of_markers * (board_size[0] * board_size[1]) * 0.5

        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.CharucoDetector(board=self.board, charucoParams=cv2.aruco.CharucoParameters(), detectorParams=detector_params)


    def get_camera_t_marker(self, images:list[np.ndarray], camera_matrix:np.ndarray, distortion_coefficients:list[float])->list[np.ndarray | None]:
        """
        :param images: NxWxHx3-uint8 RGB images
        :param camera_matrix: 3x3 camera matrix
        :param distortion_coefficients: array of the distortion coefficients of the camera
        :return: a list of 4x4 Camera^T_CharucoBoard estimates or None if an image has no aruco marker
        """

        camera_t_charuco_s = []
        for index, image in enumerate(images):
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(image)

            if len(marker_ids) < self.min_number_of_markers:
                camera_t_charuco_s.append(None)
                continue

            chessboard_obj_points, chessboard_img_points = self.board.matchImagePoints(
                charuco_corners,
                charuco_ids
            )

            marker_obj_points = np.empty((0, 3))
            marker_img_points = np.empty((0, 2))

            for i, marker_id in enumerate(marker_ids):

                marker_img_corner_s = marker_corners[i][0]

                corner_indices = list(self.board.getIds()).index(marker_id[0])
                marker_obj_corner_s = self.board.getObjPoints()[corner_indices]

                marker_obj_points = np.concatenate([marker_obj_points, marker_obj_corner_s], axis=0)
                marker_img_points = np.concatenate([marker_img_points, marker_img_corner_s], axis=0)

            # img_copy = image.copy()
            # cv2.aruco.drawDetectedCornersCharuco(img_copy, charuco_corners, charuco_ids, (0, 255, 0))
            # cv2.aruco.drawDetectedMarkers(img_copy, marker_corners, marker_ids, (0, 0, 255))
            # cv2.imshow("image",img_copy)
            # cv2.waitKey(0)

            combined_obj_points = np.concatenate([chessboard_obj_points.reshape(-1, 3), marker_obj_points], axis=0)
            combined_img_points = np.concatenate([chessboard_img_points.reshape(-1, 2), marker_img_points], axis=0)

            valid, rvec, tvec = cv2.solvePnP(
                combined_obj_points,
                combined_img_points,
                camera_matrix,
                distortion_coefficients,
            )

            if valid:
                camera_t_charuco_s.append(assemble_homogeneous_matrix(rvec=rvec, tvec=tvec))
            else:
                camera_t_charuco_s.append(None)

        return camera_t_charuco_s


