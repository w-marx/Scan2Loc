import numpy as np
import os, sys, json, cv2, shutil
from assertion_helpers import assert_intrinsic_mat, assert_homogeneous_mat

class ProtoRobotData:
    def __init__(
            self,
            depth_images:np.ndarray,
            cam_intrinsic_mtx:np.ndarray,
            cam_distortion_coefficients:list[float],
            bgr_images:np.ndarray,
            base_t_gripper_s:np.ndarray,
    ):
        assert depth_images.shape[:3] == bgr_images.shape[:3]
        assert depth_images.ndim == 3
        self._depth_images = depth_images

        assert assert_intrinsic_mat(cam_intrinsic_mtx, hxw_img=depth_images[0])
        self._cam_intrinsic_mtx = cam_intrinsic_mtx
        self._cam_distortion_coefficients = cam_distortion_coefficients

        assert bgr_images.shape[-1] == 3 and bgr_images.ndim == 4
        self._bgr_images = bgr_images

        assert base_t_gripper_s.ndim == 3
        assert all([assert_homogeneous_mat(x) for x in base_t_gripper_s])
        self._base_t_gripper_s = base_t_gripper_s

    def save(self, output_folder:str):
        """
        Creates the following output folder format:

        `output_folder`
        ├── robot
        │   └── multiple folders (000000 - min(999999, number_of_positions)) with the contents:
        │       ├──  rgb.png
        │       ├──  poses.json
        │       └──  depth.npy
        └── robot_cam_calibration.json

        robot_cam_calibration.json has the following attributes:
        - `camera_intrinsic_matrix`: the intrinsic camera matrix of the camera,
        - `camera_distortion_coefficients`: the distortion coefficients of the camera,

        :param output_folder: the name of the output folder
        """
        if os.path.exists(f"{output_folder}"):
            print(f"Output folder already exists, deleting it ...")
            shutil.rmtree(f"{output_folder}")

        camera_data = {
            'camera_intrinsic_matrix': self.cam_intrinsic_mtx.tolist(),
            'camera_distortion_coefficients': self.cam_distortion_coefficients(),
        }
        os.makedirs(f"{output_folder}", exist_ok=True)
        with open(f"{output_folder}/robot_cam_calibration.json", 'w') as f:
            json.dump(camera_data, f, indent=4)

        for i, (depth_image, bgr_image, base_t_gripper) in enumerate(zip(self.depth_images, self.bgr_images, self.base_t_gripper_s)):
            # save robot images
            os.makedirs(f"{output_folder}/robot/{i:06d}", exist_ok=True)
            cv2.imwrite(f"{output_folder}/robot/{i:06d}/rgb.png",bgr_image)
            np.save(f"{output_folder}/robot/{i:06d}/depth.npy", depth_image)

            pose_dict = {
                "base_t_gripper": base_t_gripper.tolist(),
            }
            with open(f"{output_folder}/robot/{i:06d}/poses.json", 'w') as f:
                json.dump(pose_dict, f, indent=4)

    @property
    def depth_images(self) -> np.ndarray:
        return self._depth_images

    @property
    def cam_intrinsic_mtx(self) -> np.ndarray:
        return self._cam_intrinsic_mtx

    @property
    def cam_distortion_coefficients(self) -> list[float]:
        return self._cam_distortion_coefficients

    @property
    def bgr_images(self) -> np.ndarray:
        return self._bgr_images

    @property
    def base_t_gripper_s(self) -> np.ndarray:
        return self._base_t_gripper_s

    @classmethod
    def from_folder(cls, location:str):
        print("loading data from disk for further processing ...")
        folders = sorted(os.listdir(f"{location}/robot"))

        bgr_images = [cv2.imread(f"{location}/robot/{folder}/rgb.png") for folder in folders]

        base_t_gripper_s = []

        for robot_folder in folders:
            with open(f"{location}/robot/{robot_folder}/poses.json", 'r') as f:
                base_t_gripper_s.append(np.array(json.load(f)["base_t_gripper"]))

        with open(f"{location}/robot_cam_calibration.json", 'r') as f:
            json_file = json.load(f)
            cam_mat = np.array(json_file["camera_intrinsic_matrix"])
            cam_dist_coef = json_file["camera_distortion_coefficients"]

        folders = sorted(os.listdir(f"{location}/robot"))
        depth_images = np.array([np.load(f"{location}/robot/{folder}/depth.npy") for folder in folders])

        return cls(
            depth_images=depth_images,
            cam_intrinsic_mtx=cam_mat,
            cam_distortion_coefficients=cam_dist_coef,
            bgr_images=np.array(bgr_images),
            base_t_gripper_s=np.array(base_t_gripper_s),
        )