import numpy as np

from aruco_charuco_detection import ArucoCharucoDetector
from preprocessing_2_prediction import assert_intrinsic_mat, assert_homogeneous_mat

class GatheredRobotData:
    def __init__(
        self,
        name:str,
        robot_bgr_images:np.ndarray,
        robot_bgr_cam_mtx:np.ndarray,
        robot_bgr_distortion_coefficients: list[float],
        robot_depth_images:np.ndarray,
        robot_depth_cam_mtx:np.ndarray,
        robot_depth_distortion_coefficients:list[float],
        robot_base_t_camera_s:np.ndarray,
        robot_camera_t_marker_s:list[None | np.ndarray],
        marker_detector: None | ArucoCharucoDetector
    ):
        """
        :param name: The name of the dataset (will be used for saving)
        :param robot_bgr_images: Robot BGR images as an NxHxW-uint8 numpy array
        :param robot_bgr_cam_mtx: Robot BGR camera intrinsics 3x3 matrix
        :param robot_bgr_distortion_coefficients: Robot BGR distortion coefficients, as a list of floats
        :param robot_depth_images: Robot depth images as an NxHxW-float numpy array (in meters)
        :param robot_depth_cam_mtx: Robot depth camera intrinsics 3x3 matrix
        :param robot_depth_distortion_coefficients: Robot depth distortion coefficients, as a list of floats
        :param robot_base_t_camera_s: The homogeneous 4x4 transformation base->camera matrices (Nx4x4-float numpy array)
        :param robot_camera_t_marker_s: A list of homogeneous 4x4-float camera->marker transformation matrices or None
        :param marker_detector: Aruco charuco detector instance.
        """
        self._name = name

        assert robot_bgr_images.ndim == 4, f"Wrong shape of BGR images {robot_bgr_images.shape}"
        assert robot_bgr_images.shape[0] > 0, f"No BGR images {robot_bgr_images.shape}"
        assert robot_bgr_images.dtype == np.uint8, f"Wrong dtype for bgr images {robot_bgr_images.dtype}"
        self._robot_bgr_images = robot_bgr_images

        assert assert_intrinsic_mat(robot_bgr_cam_mtx, robot_bgr_images[0])
        self._robot_bgr_cam_mtx = robot_bgr_cam_mtx

        assert isinstance(robot_bgr_distortion_coefficients, list) and len(robot_bgr_distortion_coefficients) > 0
        self._robot_bgr_distortion_coefficients = robot_bgr_distortion_coefficients

        assert robot_depth_images.shape[:3] == robot_bgr_images.shape[:3] and robot_depth_images.ndim == 3
        assert np.issubdtype(robot_depth_images.dtype, np.floating)
        self._robot_depth_images = robot_depth_images

        assert assert_intrinsic_mat(robot_depth_cam_mtx, robot_depth_images[0])
        self._robot_depth_cam_mtx = robot_depth_cam_mtx

        assert isinstance(robot_depth_distortion_coefficients, list) and len(robot_depth_distortion_coefficients) > 0
        self._robot_depth_distortion_coefficients = robot_depth_distortion_coefficients

        assert robot_base_t_camera_s.shape[0] == robot_bgr_images.shape[0]
        assert all([assert_homogeneous_mat(m) for m in robot_base_t_camera_s])
        self._robot_base_t_camera_s = robot_base_t_camera_s

        assert len(robot_camera_t_marker_s) == robot_bgr_images.shape[0]
        assert [m is None or assert_homogeneous_mat(m) for m in robot_camera_t_marker_s]
        self._robot_camera_t_marker_s = robot_camera_t_marker_s

        assert marker_detector is None or isinstance(marker_detector,ArucoCharucoDetector)
        self._marker_detector = marker_detector



    @classmethod
    def from_folder(cls,folder_path):
        """
        Takes a folder like this and creates a GatheredRobotData object
        Input data folder should have the following structure:

        `folder_path`
        ├── robot
        │   └── multiple folders
        │       ├──  rgb.png
        │       ├──  depth.npy
        │       └──  poses.json
        ├── metadata.json
        └── robot_cam_calibration.json

        The robot_cam_calibration.json file should contain the fields:
        `rgb_camera_matrix`, `depth_camera_matrix`, `rgb_distortion_coefficients` and `depth_distortion_coefficients`.`

        The `poses.json` files should contain the field:
        -`base_t_cam`
        And may contain the field:
        -`camera_t_marker`

        The metadata.json file should contain the fields to build an aruco marker detector.
        :param folder_path: the location of the input data folder
        """
        import os, cv2, json

        robot_folder_names = sorted([f"{folder_name}" for folder_name in os.listdir(f"{folder_path}/robot")])
        robot_bgr_images, robot_depth_images, robot_base_t_cameras, robot_camera_t_marker_s = [], [], [], []

        for folder in robot_folder_names:
            location = f"{folder_path}/robot/{folder}"

            robot_bgr_images.append(cv2.imread(f"{location}/rgb.png"))
            robot_depth_images.append(np.load(f"{location}/depth.npy"))

            poses_dict = json.load(open(f"{location}/poses.json"))

            robot_base_t_cameras.append(np.array(poses_dict["base_t_cam"]))
            robot_camera_t_marker_s.append(np.array(poses_dict["camera_t_marker"]) if poses_dict["camera_t_marker"] is not None else None)

        robot_cam_calibration = json.loads(open(f"{folder_path}/robot_cam_calibration.json").read())

        marker_detector = ArucoCharucoDetector.from_json(f"{folder_path}/metadata.json")

        instance = cls(
            name=os.path.dirname(folder_path),
            robot_bgr_images=np.array(robot_bgr_images),
            robot_bgr_cam_mtx=np.array(robot_cam_calibration["rgb_camera_matrix"]),
            robot_bgr_distortion_coefficients=robot_cam_calibration["rgb_distortion_coefficients"],
            robot_depth_images=np.array(robot_depth_images),
            robot_depth_cam_mtx=np.array(robot_cam_calibration["depth_camera_matrix"]),
            robot_depth_distortion_coefficients=robot_cam_calibration["depth_distortion_coefficients"],
            robot_base_t_camera_s=np.array(robot_base_t_cameras),
            robot_camera_t_marker_s=robot_camera_t_marker_s,
            marker_detector=marker_detector
        )
        return instance

    @property
    def name(self)->str:
        return self._name

    @property
    def robot_bgr_images(self) -> np.ndarray:
        return self._robot_bgr_images

    @property
    def robot_bgr_cam_mtx(self) -> np.ndarray:
        return self._robot_bgr_cam_mtx

    @property
    def robot_bgr_distortion_coefficients(self) -> list[float]:
        return self._robot_bgr_distortion_coefficients

    @property
    def robot_depth_images(self) -> np.ndarray:
        return self._robot_depth_images

    @property
    def robot_depth_cam_mtx(self) -> np.ndarray:
        return self._robot_depth_cam_mtx

    @property
    def robot_depth_distortion_coefficients(self) -> list[float]:
        return self._robot_depth_distortion_coefficients

    @property
    def robot_base_t_camera_s(self) -> np.ndarray:
        return self._robot_base_t_camera_s

    @property
    def robot_camera_t_marker_s(self) -> list[None | np.ndarray]:
        return self._robot_camera_t_marker_s

    @property
    def marker_detector(self) -> None | ArucoCharucoDetector:
        return self._marker_detector