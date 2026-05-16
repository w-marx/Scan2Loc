import os
import numpy as np

from aruco_charuco_detection import ArucoCharucoDetector
from preprocessing_2_prediction import assert_intrinsic_mat, assert_homogeneous_mat

calc_rotational_difference = lambda x, y: np.arccos(np.clip((np.trace(x[:3, :3] @ y[:3, :3].T) - 1) / 2, -1.0, 1.0))
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
    assert all([assert_homogeneous_mat(m) for m in poses])
    median_pose = np.eye(4)
    median_pose[:3,3] = min(poses, key = lambda x: sum([np.linalg.norm(x[:3,3]-y[:3,3]) for y in poses]))[:3,3]
    median_pose[:3,:3] = min(poses, key = lambda x: sum([calc_rotational_difference(x,y) for y in poses]))[:3,:3]
    return median_pose

class GatheredRobotData:
    def __init__(
        self,
        name:str,
        bgr_images:np.ndarray,
        color_cam_mtx:np.ndarray,
        color_cam_distortion_coefficients: list[float],
        depth_images:np.ndarray,
        depth_cam_mtx:np.ndarray,
        depth_cam_distortion_coefficients:list[float],
        base_t_gripper_s:np.ndarray,
        camera_t_marker_s:list[None | np.ndarray],
        marker_detector: None | ArucoCharucoDetector,
        gripper_t_cam: np.ndarray,
    ):
        """
        :param name: The name of the dataset (will be used for saving)
        :param bgr_images: Robot BGR images as an NxHxW-uint8 numpy array
        :param color_cam_mtx: Robot BGR camera intrinsics 3x3 matrix
        :param color_cam_distortion_coefficients: Robot BGR distortion coefficients, as a list of floats
        :param depth_images: Robot depth images as an NxHxW-float numpy array (in meters)
        :param depth_cam_mtx: Robot depth camera intrinsics 3x3 matrix
        :param depth_cam_distortion_coefficients: Robot depth distortion coefficients, as a list of floats
        :param base_t_gripper_s: The homogeneous 4x4 transformation base->camera matrices (Nx4x4-float numpy array)
        :param camera_t_marker_s: A list of homogeneous 4x4-float camera->marker transformation matrices or None
        :param marker_detector: Aruco charuco detector instance.
        """
        self._name = name

        assert bgr_images.ndim == 4, f"Wrong shape of BGR images {bgr_images.shape}"
        assert bgr_images.shape[0] > 0, f"No BGR images {bgr_images.shape}"
        assert bgr_images.dtype == np.uint8, f"Wrong dtype for bgr images {bgr_images.dtype}"
        self._bgr_images = bgr_images

        assert assert_intrinsic_mat(color_cam_mtx, bgr_images[0])
        self._color_cam_mtx = color_cam_mtx

        assert isinstance(color_cam_distortion_coefficients, list) and len(color_cam_distortion_coefficients) > 0
        self._color_cam_distortion_coefficients = color_cam_distortion_coefficients

        assert depth_images.shape[:3] == bgr_images.shape[:3] and depth_images.ndim == 3
        assert np.issubdtype(depth_images.dtype, np.floating)
        self._depth_images = depth_images

        assert assert_intrinsic_mat(depth_cam_mtx, depth_images[0])
        self._depth_cam_mtx = depth_cam_mtx

        assert isinstance(depth_cam_distortion_coefficients, list) and len(depth_cam_distortion_coefficients) > 0
        self._depth_cam_distortion_coefficients = depth_cam_distortion_coefficients

        assert base_t_gripper_s.shape[0] == bgr_images.shape[0]
        assert all([assert_homogeneous_mat(m) for m in base_t_gripper_s])
        self._base_t_gripper_s = base_t_gripper_s

        assert len(camera_t_marker_s) == bgr_images.shape[0]
        assert [m is None or assert_homogeneous_mat(m) for m in camera_t_marker_s]
        self._camera_t_marker_s = camera_t_marker_s

        assert marker_detector is None or isinstance(marker_detector,ArucoCharucoDetector)
        self._marker_detector = marker_detector

        assert assert_homogeneous_mat(gripper_t_cam)
        self._gripper_t_cam = gripper_t_cam



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
        ├── gripper_t_cam.npy
        ├── metadata.json
        └── robot_cam_calibration.json

        The robot_cam_calibration.json file should contain the fields:
        `rgb_camera_matrix`, `depth_camera_matrix`, `rgb_distortion_coefficients` and `depth_distortion_coefficients`.`

        The `poses.json` files should contain the field:
        -`base_t_gripper`
        And may contain the field:
        -`camera_t_marker`

        The metadata.json file should contain the fields to build an aruco marker detector.
        :param folder_path: the location of the input data folder
        """
        import os, cv2, json

        robot_folder_names = sorted([f"{folder_name}" for folder_name in os.listdir(f"{folder_path}/robot")])
        robot_bgr_images, robot_depth_images, base_t_gripper_s, robot_camera_t_marker_s = [], [], [], []

        for folder in robot_folder_names:
            location = f"{folder_path}/robot/{folder}"

            robot_bgr_images.append(cv2.imread(f"{location}/rgb.png"))
            robot_depth_images.append(np.load(f"{location}/depth.npy"))

            poses_dict = json.load(open(f"{location}/poses.json"))

            base_t_gripper_s.append(np.array(poses_dict["base_t_gripper"]))
            robot_camera_t_marker_s.append(np.array(poses_dict["camera_t_marker"]) if poses_dict["camera_t_marker"] is not None else None)

        robot_cam_calibration = json.loads(open(f"{folder_path}/robot_cam_calibration.json").read())

        marker_detector = ArucoCharucoDetector.from_json(f"{folder_path}/metadata.json")

        gripper_t_cam = np.load(f"{folder_path}/gripper_t_cam.npy")

        instance = cls(
            name=os.path.dirname(folder_path),
            bgr_images=np.array(robot_bgr_images),
            color_cam_mtx=np.array(robot_cam_calibration["rgb_camera_matrix"]),
            color_cam_distortion_coefficients=robot_cam_calibration["rgb_distortion_coefficients"],
            depth_images=np.array(robot_depth_images),
            depth_cam_mtx=np.array(robot_cam_calibration["depth_camera_matrix"]),
            depth_cam_distortion_coefficients=robot_cam_calibration["depth_distortion_coefficients"],
            base_t_gripper_s=np.array(base_t_gripper_s),
            camera_t_marker_s=robot_camera_t_marker_s,
            marker_detector=marker_detector,
            gripper_t_cam=gripper_t_cam
        )
        return instance


    def save(self, folder, new_name=None):
        import shutil, cv2, json
        location = f"{folder}/{self.name}" if new_name is None else f"{folder}/{new_name}"

        if os.path.exists(location):
            print(f"Output folder already exists, deleting it ...")
            shutil.rmtree(location)

        padding = len(str(self.bgr_images.shape[0]-1))
        for i in range(self.bgr_images.shape[0]):
            robot_folder = f"{location}/robot/{str(i).zfill(padding)}"
            os.makedirs(robot_folder, exist_ok=True)

            cv2.imwrite(f"{robot_folder}/rgb.png", self.bgr_images[i])
            np.save(f"{robot_folder}/depth.npy", self.depth_images[i])

            poses_dict = {
                "base_t_gripper": self.base_t_gripper_s[i].tolist(),
                "camera_t_marker": self.camera_t_marker_s[i].tolist() if self.camera_t_marker_s[i] is not None else None,
            }
            with open(f"{robot_folder}/poses.json", 'w') as f:
                json.dump(poses_dict, f, indent=4)

        with open(f"{location}/metadata.json", 'w') as f:
            json.dump(self.marker_detector.get_meta_data() if self.marker_detector is not None else None, f, indent=4)

        robot_cam_calibration = {
            "rgb_camera_matrix":self.color_cam_mtx.tolist(),
            "rgb_distortion_coefficients":self.color_distortion_coefficients,
            "depth_camera_matrix":self.depth_cam_mtx.tolist(),
            "depth_distortion_coefficients":self.depth_cam_distortion_coefficients
        }
        with open(f"{location}/robot_cam_calibration.json", 'w') as f:
            json.dump(robot_cam_calibration, f, indent=4)

        np.save(f"{location}/gripper_t_cam.npy", self._gripper_t_cam)


    @property
    def name(self)->str:
        return self._name

    @property
    def bgr_images(self) -> np.ndarray:
        return self._bgr_images

    @property
    def color_cam_mtx(self) -> np.ndarray:
        return self._color_cam_mtx

    @property
    def color_distortion_coefficients(self) -> list[float]:
        return self._color_cam_distortion_coefficients

    @property
    def depth_images(self) -> np.ndarray:
        return self._depth_images

    @property
    def depth_cam_mtx(self) -> np.ndarray:
        return self._depth_cam_mtx

    @property
    def depth_cam_distortion_coefficients(self) -> list[float]:
        return self._depth_cam_distortion_coefficients

    @property
    def base_t_gripper_s(self) -> np.ndarray:
        return self._base_t_gripper_s

    @property
    def gripper_t_cam(self) -> np.ndarray:
        return self._gripper_t_cam

    @property
    def base_t_camera_s(self) -> np.ndarray:
        return np.array([b_t_g @ self.gripper_t_cam for b_t_g in self.base_t_gripper_s])

    @property
    def camera_t_marker_s(self) -> list[None | np.ndarray]:
        return self._camera_t_marker_s

    @property
    def marker_detector(self) -> None | ArucoCharucoDetector:
        return self._marker_detector