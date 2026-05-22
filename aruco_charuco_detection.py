import numpy as np
import cv2


ARUCO_DICTIONARY_OPTIONS = {
    "4X4_250": cv2.aruco.DICT_4X4_250,
    "5X5_100": cv2.aruco.DICT_5X5_100,
    "5X5_250": cv2.aruco.DICT_5X5_250,
    "6X6_250": cv2.aruco.DICT_6X6_250,
    "7X7_250": cv2.aruco.DICT_7X7_250,
    "7X7_1000": cv2.aruco.DICT_7X7_1000,
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



class ArucoCharucoDetector:
    def __init__(self):
        self.marker_remover = LamaMasker()
    
    def set_new_masker(self, masker:str):
        if masker == "ImageMasker":
            self.marker_remover = ImageMasker()
        elif masker == "LamaMasker":
            self.marker_remover = LamaMasker()
        else:
            raise Exception(f"Masker {masker} not found")

    def get_camera_t_marker(self, images:list[np.ndarray], camera_matrix:np.ndarray, distortion_coefficients:list[float]|None = None)->list[np.ndarray | None]:
        """
        Returns the pose camera_t_marker or for each image in the list as a list of 4x4 homogeneous matrices
        :param images: list of HxWx3 RGB images
        :param camera_matrix: the 3x3 intrinsic camera matrix
        :param distortion_coefficients: the distortion coefficients of the camera
        :return: list of 4x4 homogeneous matrices
        """
        raise Exception("ArucoCharucoDetector is no concrete class - pose estimation function not implemented")

    def remove_markers(self, images:list[np.ndarray])->list[np.ndarray]:
        """
        Returns the images with the markers digitally removed (pixels set to 0)
        :param images: list of HxWx3 RGB images
        :return: list of WxHx3 RGB images without the aruco markers
        """
        raise Exception("ArucoCharucoDetector is no concrete class - marker removal function not implemented")

    def get_meta_data(self):
        """
        Returns some metadata about the detection process
        """
        return {}

    @classmethod
    def from_json(cls, json_file:str):
        """
        Builds an aruco charuco detector from the json file
        :param json_file: the location of the json file
        :return: an aruco/charuco Detector or None
        """
        import json, os
        if not os.path.exists(json_file):
            print(f"No detector creation json found, returning None {json_file}")
            return None

        metadata_dict = json.load(open(json_file))

        if metadata_dict["Aruco/Charuco Type"] is None:
            return None
        if metadata_dict["Aruco/Charuco Type"] == "Aruco":
            return ArucoDetector(
                aruco_marker_side_length=metadata_dict["Aruco marker side length"],
                aruco_marker_dictionary=metadata_dict["Aruco dictionary"]
            )
        elif metadata_dict["Aruco/Charuco Type"] == "Charuco":
            return CharucoDetector(
                board_size=(metadata_dict["Charuco board size"][0], metadata_dict["Charuco board size"][1]),
                square_size=metadata_dict["Charuco square size"],
                marker_size=metadata_dict["Aruco marker side length"],
                aruco_dictionary=metadata_dict["Aruco dictionary"],
                min_fraction_of_markers=metadata_dict["Min fraction of markers"]
            )
        else:
            raise Exception("Unknown aruco charuco type")


class ArucoDetector(ArucoCharucoDetector):
    def __init__(
            self,
            aruco_marker_side_length:float,
            aruco_marker_dictionary:str = "5X5_250",
    ):
        super().__init__()
        self.meta_data = {
            "Aruco/Charuco Type":"Aruco",
            "Aruco marker side length": aruco_marker_side_length,
            "Aruco dictionary": aruco_marker_dictionary,
        }

        self.aruco_marker_side_length = aruco_marker_side_length
        self.aruco_marker_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_OPTIONS[aruco_marker_dictionary])
        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(self.aruco_marker_dictionary, detector_params)

    def get_camera_t_marker(self, images:list[np.ndarray], camera_matrix:np.ndarray, distortion_coefficients:list[float]|None = None)->list[np.ndarray | None]:

        marker_points = np.array([[-1,1,0], [1,1,0], [1,-1,0], [-1,-1,0]])*0.5*self.aruco_marker_side_length

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

    def get_meta_data(self):
        return self.meta_data


class CharucoDetector(ArucoCharucoDetector):
    def __init__(
            self,
            board_size:tuple[int,int] = (14, 9),
            square_size:float = 0.0188,
            marker_size:float = 0.0146,
            aruco_dictionary:str = "5X5_250",
            min_fraction_of_markers:float = 1.0
    ):
        super().__init__()
        self.square_size = square_size
        self.marker_size = marker_size
        self.board = cv2.aruco.CharucoBoard(board_size, square_size, marker_size, cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY_OPTIONS[aruco_dictionary]))
        self.min_number_of_markers = min_fraction_of_markers * (board_size[0] * board_size[1]) * 0.5

        detector_params = cv2.aruco.DetectorParameters()
        detector_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.CharucoDetector(board=self.board, charucoParams=cv2.aruco.CharucoParameters(), detectorParams=detector_params)


        self.metadata = {
            "Aruco/Charuco Type":"Charuco",
            "Aruco marker side length": marker_size,
            "Charuco square size": square_size,
            "Aruco dictionary": aruco_dictionary,
            "Min fraction of markers": min_fraction_of_markers,
            "Charuco board size": [board_size[0], board_size[1]]
        }


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
                marker_square = marker_avg+((marker-marker_avg)*self.square_size/self.marker_size)*2
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
    
    def get_meta_data(self):
        return self.metadata

if __name__ == "__main__":
    import os
    image_folder = "../datasets/charuco1/robot"
    if not os.path.exists(image_folder):
        Exception("Folder {image_folder} does not exist")
    image_paths = [f"{image_folder}/{folder}/rgb.png" for folder in os.listdir(image_folder)]
    images = [cv2.imread(name) for name in image_paths]
    charuco_detector = CharucoDetector()
    charuco_detector.remove_markers(images)