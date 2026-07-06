import os
import numpy as np

from shared.aruco_charuco_detection import MarkerDetector
from shared.assertion_helpers import *

class CompleteRobotScan:
    def __init__(
        self,
        name:str,
        bgr_images:np.ndarray,
        cam_intrinsic_mtx:np.ndarray,
        depth_images:np.ndarray,
        base_t_gripper_s:np.ndarray,
        camera_t_marker_s:list[None | np.ndarray],
        marker_detector: MarkerDetector,
        gripper_t_cam: np.ndarray,
    ):
        """
        :param name: The name of the dataset (will be used for saving)
        :param bgr_images: Robot BGR images as an NxHxW-uint8 numpy array
        :param cam_intrinsic_mtx: Robot BGR camera intrinsics 3x3 matrix
        :param depth_images: Robot depth images as an NxHxW-float numpy array (in meters)
        :param base_t_gripper_s: The homogeneous 4x4 transformation base->camera matrices (Nx4x4-float numpy array)
        :param camera_t_marker_s: A list of homogeneous 4x4-float camera->marker transformation matrices or None
        :param marker_detector: Aruco charuco detector instance.
        """
        self._name = name

        assert bgr_images.ndim == 4, f"Wrong shape of BGR images {bgr_images.shape}"
        assert bgr_images.shape[0] > 0, f"No BGR images {bgr_images.shape}"
        assert bgr_images.dtype == np.uint8, f"Wrong dtype for bgr images {bgr_images.dtype}"
        self._bgr_images = bgr_images

        assert assert_intrinsic_mat(cam_intrinsic_mtx, bgr_images[0])
        self._cam_intrinsic_mtx = cam_intrinsic_mtx

        assert depth_images.shape[:3] == bgr_images.shape[:3] and depth_images.ndim == 3
        assert np.issubdtype(depth_images.dtype, np.floating)
        self._depth_images = depth_images

        assert base_t_gripper_s.shape[0] == bgr_images.shape[0]
        assert all([assert_homogeneous_mat(m) for m in base_t_gripper_s])
        self._base_t_gripper_s = base_t_gripper_s

        assert len(camera_t_marker_s) == bgr_images.shape[0]
        assert [m is None or assert_homogeneous_mat(m) for m in camera_t_marker_s]
        self._camera_t_marker_s = camera_t_marker_s

        assert isinstance(marker_detector,MarkerDetector)
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
        │       ├──  depth.npz
        │       └──  poses.json
        ├── gripper_t_cam.npy
        ├── marker_detector_config.json
        └── robot_cam_calibration.json

        The robot_cam_calibration.json file should contain the fields:
        `camera_intrinsic_matrix`

        The `poses.json` files should contain the field:
        -`base_t_gripper`
        And may contain the field:
        -`camera_t_marker`

        The marker_detector_config.json file should contain the fields to build an aruco marker detector.
        :param folder_path: the location of the input data folder
        """
        import os, cv2, json

        robot_folder_names = sorted([f"{folder_name}" for folder_name in os.listdir(f"{folder_path}/robot")])
        robot_bgr_images, robot_depth_images, base_t_gripper_s, robot_camera_t_marker_s = [], [], [], []

        for folder in robot_folder_names:
            location = f"{folder_path}/robot/{folder}"

            robot_bgr_images.append(cv2.imread(f"{location}/rgb.png"))
            robot_depth_images.append(np.load(f"{location}/depth.npz")['depth'])

            poses_dict = json.load(open(f"{location}/poses.json"))

            base_t_gripper_s.append(np.array(poses_dict["base_t_gripper"]))
            robot_camera_t_marker_s.append(np.array(poses_dict["camera_t_marker"]) if poses_dict["camera_t_marker"] is not None else None)

        robot_cam_calibration = json.loads(open(f"{folder_path}/robot_cam_calibration.json").read())

        with open(f"{folder_path}/marker_detector_config.json") as f:
            marker_config_dict = json.load(f)
        marker_detector = MarkerDetector.from_dict(marker_config_dict)

        gripper_t_cam = np.load(f"{folder_path}/gripper_t_cam.npy")

        instance = cls(
            name=os.path.dirname(folder_path),
            bgr_images=np.array(robot_bgr_images),
            cam_intrinsic_mtx=np.array(robot_cam_calibration["camera_intrinsic_matrix"]),
            depth_images=np.array(robot_depth_images),
            base_t_gripper_s=np.array(base_t_gripper_s),
            camera_t_marker_s=robot_camera_t_marker_s,
            marker_detector=marker_detector,
            gripper_t_cam=gripper_t_cam
        )
        return instance


    def save(self, folder:str, new_name:str=None, dist_coeff:list[float] = None):
        import shutil, cv2, json
        location = f"{folder}/{self.name}" if new_name is None else f"{folder}/{new_name}"

        if os.path.exists(location):
            print(f"Output folder already exists, deleting it ...")
            shutil.rmtree(location)

        padding = len(str(self.bgr_images.shape[0]-1))
        for i in range(self.bgr_images.shape[0]):
            robot_folder = f"{location}/robot/{str(i).zfill(padding)}"
            os.makedirs(robot_folder, exist_ok=True)

            cv2.imwrite(
                f"{robot_folder}/rgb.png", 
                self.bgr_images[i],
                [cv2.IMWRITE_PNG_COMPRESSION, 9]
            )
            np.savez_compressed(f"{robot_folder}/depth.npz", depth = self.depth_images[i].astype(np.float32))

            poses_dict = {
                "base_t_gripper": self.base_t_gripper_s[i].tolist(),
                "camera_t_marker": self.camera_t_marker_s[i].tolist() if self.camera_t_marker_s[i] is not None else None,
            }
            with open(f"{robot_folder}/poses.json", 'w') as f:
                json.dump(poses_dict, f, indent=4)

        with open(f"{location}/marker_detector_config.json", 'w') as f:
            json.dump(self.marker_detector.config_dict, f, indent=4)

        robot_cam_calibration = {"camera_intrinsic_matrix":self.cam_intrinsic_mtx.tolist()}
        if dist_coeff is not None:
            robot_cam_calibration.update({'camera_distortion_coefficients': dist_coeff})

        with open(f"{location}/robot_cam_calibration.json", 'w') as f:
            json.dump(robot_cam_calibration, f, indent=4)

        np.save(f"{location}/gripper_t_cam.npy", self._gripper_t_cam)
    
    def see_color_depth_alignment(self,frame_idx:int = 0):
        assert frame_idx >= 0
        if frame_idx >= self.bgr_images.shape[0]:
            return
        bgr_image = self.bgr_images[frame_idx].copy()
        depth_image = self.depth_images[frame_idx].copy()

        import matplotlib.pyplot as plt
        import cv2

        gray_bgr = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    
        # Depth image as color gradient
        depth_image_norm = (depth_image - np.min(depth_image))/(np.max(depth_image) - np.min(depth_image)+0.001)
        depth_gradient = plt.cm.jet(depth_image_norm)[:,:, :3]
        
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        alpha = 0.6 
        ax1.imshow(depth_gradient * alpha + (np.stack([gray_bgr/255.0]*3, axis=2)) * (1-alpha))
        ax1.set_title('Grayscale BGR + Gradient Depth')

        ax2.imshow(depth_gradient)
        ax2.set_title('Only depth gradient')
        plt.show()


    @property
    def name(self)->str:
        return self._name

    @property
    def bgr_images(self) -> np.ndarray:
        return self._bgr_images

    @property
    def cam_intrinsic_mtx(self) -> np.ndarray:
        return self._cam_intrinsic_mtx

    @property
    def depth_images(self) -> np.ndarray:
        return self._depth_images

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
    def base_t_marker_s(self) -> list[None | np.ndarray]:
        return [
            (None if c_t_m is None else b_t_c @ c_t_m) 
            for b_t_c,c_t_m in zip(self.base_t_camera_s, self._camera_t_marker_s)
        ]

    @property
    def marker_detector(self) -> None | MarkerDetector:
        return self._marker_detector