import numpy as np
import cv2
from dataclasses import dataclass, asdict
from typing import Literal
from abc import ABC, abstractmethod


ARUCO_DICTIONARY_OPTIONS = {
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "7X7_250": cv2.aruco.DICT_7X7_250,
    "7X7_1000": cv2.aruco.DICT_7X7_1000,
}

@dataclass(frozen=True, kw_only=True)
class MarkerDetectionConfig:
    """
    Specifies the marker type to build an marker detector
    :param marker_type: 'Aruco'/'Charuco' or None are currently supported
    :param marker_side_length: The side length of the marker in meters
    :param aruco_marker_dictionary: The dictionary of the Aruco marker, as a string of the form "MxM_N", e.g. "5X5_250"
    :param board_size: The number of squares along each axis for charuco boards
    :param min_fraction_of_markers: The min fraction of markers on a charuco board to return a prediction
    """
    marker_type:Literal["Aruco", "Charuco"] | None = "Aruco"
    marker_side_length:float = 0.0725
    aruco_marker_dictionary:str = "5X5_250"
    board_size:list[int] | None = None
    square_size:float|None = None
    min_fraction_of_markers:float=1.0

    def __post_init__(self):
        assert self.marker_type is None or self.marker_type in ["Aruco", "Charuco"], f"unknown marker type: {self.marker_type}"
        if self.marker_type is None:
            return
        assert self.marker_side_length > 0, f"Marker side length must be positive, is: {self.marker_side_length}"
        assert self.aruco_marker_dictionary in ARUCO_DICTIONARY_OPTIONS, f"{self.aruco_marker_dictionary} is not known, supported: {ARUCO_DICTIONARY_OPTIONS.keys()}"
        if self.marker_type == "Charuco":
            assert self.board_size is not None and len(self.board_size) == 2 and self.board_size[0] > 0 and self.board_size[1] > 0, f"Unsupported Charuco board size: {self.board_size}"
            assert self.square_size is not None and self.square_size > 0, f"Invalid square size for charuco: {self.square_size}"
            assert 0 < self.min_fraction_of_markers <= 1.0, f"Fraction of markers must be in (0,1], is: {self.min_fraction_of_markers}"

DEFAULT_MARKER_CONFIGS = {
    "Aruco 5x5_250 72.5mm": MarkerDetectionConfig(),
    "Aruco 5x5_250 108.5mm": MarkerDetectionConfig(marker_side_length=0.1085),
    "Charuco 14x9 5x5_250 14.6mm 18.8mm": MarkerDetectionConfig(
        marker_type="Charuco", marker_side_length=0.0146, aruco_marker_dictionary="5X5_250", board_size=(14, 9), min_fraction_of_markers=1.0, square_size=0.0188
    ),
    "No marker": MarkerDetectionConfig(
        marker_type=None
    )
}


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

class ImageMasker:
    def remove_area(self,bgr_images:np.ndarray, hulls:list[np.ndarray]):
        """
        Paints the area inside the hulls black
        :param images an NxHxWx3-uint8 numpy array of BGR images
        :param hulls: A list of Mx2 array of image coordinates that form a hull, or None that has length N
        """
        assert bgr_images.ndim == 4 and bgr_images.shape[0] > 0
        assert len(hulls) == bgr_images.shape[0]
        assert all([hull is None or (hull.ndim == 2 and hull.shape[0] > 0 and hull.shape[-1] == 2) for hull in hulls])

        edited_images = []
        for bgr_image, hull in zip(bgr_images, hulls):
            img_copy = bgr_image.copy()
            if hull is not None:
                cv2.fillPoly(img_copy, [hull.astype(np.int32)], color=(0, 0, 0))
            edited_images.append(img_copy)
        return np.array(edited_images)


class LamaMasker(ImageMasker):
    def __init__(self):
        from simple_lama_inpainting import SimpleLama
        self.model = SimpleLama()

    def remove_area(self, bgr_images:np.ndarray, hulls:list[np.ndarray]):
        assert bgr_images.ndim == 4 and bgr_images.shape[0] > 0
        assert len(hulls) == bgr_images.shape[0]
        assert all([hull is None or (hull.ndim == 2 and hull.shape[0] > 0 and hull.shape[-1] == 2) for hull in hulls])

        masked_images = []
        for bgr_img, hull_points in zip(bgr_images, hulls):
            mask = np.zeros(bgr_img.shape[:2], dtype=np.uint8)
            if hull_points is not None:
                cv2.fillPoly(mask, [hull_points.astype(np.int32)], 255)
                cv2.polylines(mask, [hull_points.astype(np.int32)], isClosed=True, color=255, thickness=5)
            result = self.model(bgr_img, mask)
            masked_images.append(result)
        return np.array(masked_images)



class MarkerDetector(ABC):
    def __init__(self, config:MarkerDetectionConfig):
        self.marker_remover = ImageMasker()
        self._config = config
    
    def set_new_masker(self, masker:Literal["ImageMasker", "LamaMasker"]):
        assert masker in ["ImageMasker", "LamaMasker"], f"masker: {masker} not known, known: ImageMasker, LamaMasker"
        if masker == "ImageMasker":
            self.marker_remover = ImageMasker()
        elif masker == "LamaMasker":
            self.marker_remover = LamaMasker()

    @abstractmethod
    def get_camera_t_marker(self, images:list[np.ndarray], camera_matrix:np.ndarray, distortion_coefficients:list[float]|None = None)->list[np.ndarray | None]:
        """
        Returns the pose camera_t_marker or for each image in the list as a list of 4x4 homogeneous matrices
        :param images: list of HxWx3 RGB images
        :param camera_matrix: the 3x3 intrinsic camera matrix
        :param distortion_coefficients: the distortion coefficients of the camera
        :return: list of 4x4 homogeneous matrices
        """
        raise NotImplementedError("ArucoCharucoDetector is an abstract base class")

    @abstractmethod
    def remove_markers(self, images:list[np.ndarray])->list[np.ndarray]:
        """
        Returns the images with the markers digitally removed (pixels set to 0)
        :param images: list of HxWx3 RGB images
        :return: list of WxHx3 RGB images without the aruco markers
        """
        raise NotImplementedError("ArucoCharucoDetector is an abstract base class")

    @property
    def config(self)->MarkerDetectionConfig:
        """
        Returns the MarkerDetectionConfig to recreate the class
        """
        return self._config
    
    @property
    def config_dict(self)->dict:
        """
        Returns the MarkerDetectionConfig as a dictionary
        """
        return asdict(self._config)
    
    @staticmethod
    def from_dict(data:dict)->'MarkerDetector':
        return MarkerDetector.from_config(MarkerDetectionConfig(**dict))

    @classmethod
    def from_config(cls,config:MarkerDetectionConfig)->'MarkerDetector':
        if config.marker_type is None:
            return NoMarkerDetector(config)
        if config.marker_type == "Aruco":
            return ArucoDetector(config)
        if config.marker_type == "Charuco":
            return CharucoDetector(config)
        raise Exception("Unknown marker detector config")

class NoMarkerDetector(MarkerDetector):
    def __init__(self, config:MarkerDetectionConfig):
        super().__init__(config)

    def get_camera_t_marker(self, images:list[np.ndarray], camera_matrix:np.ndarray, distortion_coefficients:list[float]|None = None)->list[np.ndarray | None]:
        return [None]*len(images)
    
    def remove_markers(self, images:list[np.ndarray])->list[np.ndarray]:
        return images


class ArucoDetector(MarkerDetector):
    def __init__(self,config:MarkerDetectionConfig):
        super().__init__(config)

        self.aruco_marker_dictionary = cv2.aruco.getPredefinedDictionary(
            ARUCO_DICTIONARY_OPTIONS[self.config.aruco_marker_dictionary]
        )
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.aruco_marker_dictionary, detector_params)

    def get_camera_t_marker(self, images:list[np.ndarray], camera_matrix:np.ndarray, distortion_coefficients:list[float]|None = None)->list[np.ndarray | None]:

        marker_points = np.array([[-1,1,0], [1,1,0], [1,-1,0], [-1,-1,0]])*0.5*self.config.marker_side_length

        camera_t_aruco_s = []
        for index, image in enumerate(images):
            marker_corners, marker_ids, reject_candidates = self.detector.detectMarkers(image)

            if len(marker_corners) > 1:
                raise Exception("More then one aruco marker detected, single pose is not calculatable")
            if len(marker_corners) == 0:
                camera_t_aruco_s.append(None)
                continue

            _, rvec, tvec = cv2.solvePnP(
                objectPoints=marker_points,
                imagePoints=marker_corners[0][0],
                cameraMatrix=camera_matrix,
                distCoeffs=np.array(([0,0,0,0,0] if distortion_coefficients is None else distortion_coefficients)),
                flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            camera_t_aruco_s.append(assemble_homogeneous_matrix(rvec=rvec, tvec=tvec))

            #img_copy = image.copy()
            #cv2.aruco.drawDetectedMarkers(img_copy, marker_corners, marker_ids, (0, 0, 255))
            #cv2.imshow(f"image: {index}",img_copy)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        return camera_t_aruco_s

    def remove_markers(self, images:list[np.ndarray]) ->list[np.ndarray]:
        masked_images = []
        for image in images:
            marker_corners, marker_ids, reject_candidates = self.detector.detectMarkers(image)
            image_copy = image.copy()
            if marker_corners:
                for corners in marker_corners:
                    image_copy = self.marker_remover.remove_area(np.array([image_copy]), [corners.reshape(4,2).astype(np.int32)])[0]
            masked_images.append(image_copy)
        return masked_images


class CharucoDetector(MarkerDetector):
    def __init__(self,config:MarkerDetectionConfig):
        super().__init__(config=config)
        self.board = cv2.aruco.CharucoBoard(
            config.board_size, config.square_size, config.marker_side_length, cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_OPTIONS[config.aruco_marker_dictionary])
        )
        self.min_number_of_markers = config.min_fraction_of_markers * (config.board_size[0] * config.board_size[1]) * 0.5

        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.CharucoDetector(board=self.board, charucoParams=cv2.aruco.CharucoParameters(), detectorParams=detector_params)

    def get_camera_t_marker(self, images:list[np.ndarray], camera_matrix:np.ndarray, distortion_coefficients:list[float]|None = None)->list[np.ndarray | None]:
        """
        :param images: NxWxHx3-uint8 RGB images
        :param camera_matrix: 3x3 camera matrix
        :param distortion_coefficients: array of the distortion coefficients of the camera
        :return: a list of 4x4 Camera^T_CharucoBoard estimates or None if an image has no aruco marker
        """

        camera_t_charuco_s = []
        for index, image in enumerate(images):
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(image)

            if marker_ids is None or len(marker_ids) < self.min_number_of_markers:
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

            #img_copy = image.copy()
            #cv2.aruco.drawDetectedCornersCharuco(img_copy, charuco_corners, charuco_ids, (0, 255, 0))
            #cv2.aruco.drawDetectedMarkers(img_copy, marker_corners, marker_ids, (0, 0, 255))
            #cv2.imshow(f"image: {index}",img_copy)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            combined_obj_points = np.concatenate([chessboard_obj_points.reshape(-1, 3), marker_obj_points], axis=0)
            combined_img_points = np.concatenate([chessboard_img_points.reshape(-1, 2), marker_img_points], axis=0)

            valid, rvec, tvec = cv2.solvePnP(
                combined_obj_points,
                combined_img_points,
                camera_matrix,
                np.array(([0,0,0,0,0] if distortion_coefficients is None else distortion_coefficients)),
            )

            if valid:
                camera_t_charuco_s.append(assemble_homogeneous_matrix(rvec=rvec, tvec=tvec))
            else:
                camera_t_charuco_s.append(None)

        return camera_t_charuco_s

    def remove_markers(self, images:list[np.ndarray], advanced:bool = True)->list[np.ndarray]:
        hulls = []
        for image in images:
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(image)

            marker_squares = []
            for marker in marker_corners:
                marker = marker[0]
                marker_avg = np.mean(marker, axis=0)
                marker_square = marker_avg+((marker-marker_avg)*self.config.square_size/self.config.marker_side_length)*2
                marker_squares.append(marker_square)
            if len(marker_squares) == 0:
                hulls.append(None)
                continue

            marker_square_points = np.vstack(marker_squares)
            from scipy.spatial import ConvexHull
            hull = ConvexHull(marker_square_points)
            hull_points = marker_square_points[hull.vertices]
            hulls.append(hull_points)

        return self.marker_remover.remove_area(np.array(images), hulls)