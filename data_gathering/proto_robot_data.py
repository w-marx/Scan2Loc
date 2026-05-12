import numpy as np
import os, sys, json, cv2, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_2_prediction import assert_intrinsic_mat, assert_homogeneous_mat

class ProtoRobotData:
    def __init__(
            self,
            depth_images:np.ndarray,
            depth_cam_intrinsic_mtx:np.ndarray,
            depth_cam_distortion_coefficients:list[float],
            bgr_images:np.ndarray,
            color_cam_intrinsic_mtx:np.ndarray,
            color_cam_distortion_coefficients:list[float],
            base_t_gripper_s:np.ndarray,
    ):
        assert depth_images.shape[:3] == bgr_images.shape[:3]
        assert depth_images.ndim == 3
        self._depth_images = depth_images

        assert assert_intrinsic_mat(depth_cam_intrinsic_mtx, hxw_img=depth_images[0])
        self._depth_cam_intrinsic_mtx = depth_cam_intrinsic_mtx
        self._depth_cam_distortion_coefficients = depth_cam_distortion_coefficients

        assert bgr_images.shape[-1] == 3 and bgr_images.ndim == 4
        self._bgr_images = bgr_images

        assert assert_intrinsic_mat(color_cam_intrinsic_mtx, hxw_img=bgr_images[0])
        self._color_cam_intrinsic_mtx = color_cam_intrinsic_mtx
        self._color_cam_distortion_coefficients = color_cam_distortion_coefficients

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
        - `rgb_camera_matrix`: the intrinsic camera matrix of the rgb camera,
        - `rgb_distortion_coefficients`: the distortion coefficients of the rgb camera,
        - `depth_camera_matrix`: the intrinsic camera matrix of the depth camera,
        - `depth_distortion_coefficients`: the distortion coefficients of the depth camera,

        :param output_folder: the name of the output folder

        :return: rgb_images, base_t_gripper_s, rgb_cam_mat, rgb_cam_dist_coef
        """
        if os.path.exists(f"{output_folder}"):
            print(f"Output folder already exists, deleting it ...")
            shutil.rmtree(f"{output_folder}")

        camera_data = {
            'rgb_camera_matrix': self.color_cam_intrinsic_mtx.tolist(),
            'rgb_distortion_coefficients': self.color_cam_distortion_coefficients,
            'depth_camera_matrix': self.depth_cam_intrinsic_mtx.tolist(),
            'depth_distortion_coefficients': self.depth_cam_distortion_coefficients,
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
    def depth_cam_intrinsic_mtx(self) -> np.ndarray:
        return self._depth_cam_intrinsic_mtx

    @property
    def depth_cam_distortion_coefficients(self) -> list[float]:
        return self._depth_cam_distortion_coefficients

    @property
    def bgr_images(self) -> np.ndarray:
        return self._bgr_images

    @property
    def color_cam_intrinsic_mtx(self) -> np.ndarray:
        return self._color_cam_intrinsic_mtx

    @property
    def color_cam_distortion_coefficients(self) -> list[float]:
        return self._color_cam_distortion_coefficients

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
            color_cam_mat = np.array(json_file["rgb_camera_matrix"])
            color_cam_dist_coef = json_file["rgb_distortion_coefficients"]

        folders = sorted(os.listdir(f"{location}/robot"))
        depth_images = np.array([np.load(f"{location}/robot/{folder}/depth.npy") for folder in folders])

        with open(f"{location}/robot_cam_calibration.json", 'r') as f:
            json_file = json.load(f)
            depth_cam_mat = np.array(json_file["depth_camera_matrix"])
            depth_cam_dist_coef = json_file["depth_distortion_coefficients"]

        return cls(
            depth_images=depth_images,
            depth_cam_intrinsic_mtx=depth_cam_mat,
            depth_cam_distortion_coefficients=depth_cam_dist_coef,
            bgr_images=np.array(bgr_images),
            color_cam_intrinsic_mtx=color_cam_mat,
            color_cam_distortion_coefficients=color_cam_dist_coef,
            base_t_gripper_s=np.array(base_t_gripper_s),
        )