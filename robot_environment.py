import cv2
import numpy as np
import os, shutil, json
import open3d as o3d
from shared_utilities import assert_intrinsic_mat, assert_homogeneous_mat, create_3d_camera, assert_mxnx3_np_uint8_image_batch


class RobotEnvironment:
    def __init__(
            self,
            name:str,
            robot_bgr_images:np.ndarray,
            robot_bgr_intrinsics:np.ndarray,
            robot_xyz_images:np.ndarray,
            robot_base_t_robot_camera_s:np.ndarray,
    ):
        """
        :param name: the name of the dataset (will be stored under it)
        :param robot_bgr_images: BGR images of the robot as a NxHxWx3-uint8 numpy array
        :param robot_bgr_intrinsics: Intrinsics BGR camera matrix of the robot (3x3 numpy array)
        :param robot_xyz_images: XYZ images from the pov of the robot as a NxHxWx3-float numpy array
        :param robot_base_t_robot_camera_s: The homogeneous robot_base->robot_camera transformation matrix as a Nx4x4-float numpy array
        """
        self._name = name

        assert assert_mxnx3_np_uint8_image_batch(robot_bgr_images)
        self._robot_bgr_images = robot_bgr_images

        assert assert_intrinsic_mat(robot_bgr_intrinsics, robot_bgr_images[0])
        self._robot_bgr_intrinsics = robot_bgr_intrinsics

        assert robot_xyz_images.shape == robot_bgr_images.shape
        assert np.issubdtype(robot_xyz_images.dtype, np.floating)
        self._robot_xyz_images = robot_xyz_images

        assert robot_base_t_robot_camera_s.shape == (robot_bgr_images.shape[0],4,4), f"Wrong shape of robot_base_t_robot_camera_s {robot_base_t_robot_camera_s.shape}"
        assert all([assert_homogeneous_mat(m) for m in robot_base_t_robot_camera_s])
        self._robot_base_t_robot_camera_s = robot_base_t_robot_camera_s


    @classmethod
    def from_folder(cls, load_folder:str):
        """
        Loads from a folder of the structure:
        `output_folder`
        ├── robot_cam_calibration.json
        ├── robot
        │   └── 00 to number of datapoints
        │       ├── A xyz.npy file with the world points associated to each pixel
        │       ├── A robot_base_t_robot_camera.json with the 4x4 transformation matrix between robot base and camera
        │       └── A rgb.png image with possible aruco markers digitally removed
        └── point_cloud.ply
        """

        robot_cam_cal = json.load(open(f"{load_folder}/robot_cam_calibration.json"))

        robot_folders = sorted([f"{folder}" for folder in os.listdir(f"{load_folder}/robot")])
        robot_folders = [f"{load_folder}/robot/{folder}" for folder in robot_folders]

        robot_bgr_images = np.array([cv2.imread(f"{folder}/rgb.png") for folder in robot_folders])
        robot_xyz_images = np.array([np.load(f"{folder}/xyz.npy") for folder in robot_folders])
        robot_base_t_robot_cam_s = np.array([json.load(open(f"{folder}/robot_base_t_robot_camera.json")) for folder in robot_folders])

        instance = cls(
            name = os.path.basename(load_folder),
            robot_bgr_images = robot_bgr_images,
            robot_bgr_intrinsics = np.array(robot_cam_cal["intrinsic_camera_matrix"]),
            robot_xyz_images=robot_xyz_images,
            robot_base_t_robot_camera_s=robot_base_t_robot_cam_s,
        )
        return instance

    def save(self, folder_path:str, new_name:str|None=None):
        """
        Saves the instance to a folder, in the format that it can be recreated using `from_folder`.
        Will delete a folder if `folder_path/name` already exists.
        :param folder_path: The folder in which the instance should be saved
        :param new_name: The new name of the output_folder if not they will just use the name
        """
        location = f"{folder_path}/{new_name if new_name is not None else self.name}"

        if os.path.exists(location):
            print(f"Output folder already exists, deleting it ...")
            shutil.rmtree(location)
        os.makedirs(name=location, exist_ok=True)
        
        with open(f"{location}/robot_cam_calibration.json", 'w') as f:
            json.dump({'intrinsic_camera_matrix': self.robot_bgr_intrinsics.tolist()}, f, indent=4)

        padding = len(str(self.robot_bgr_images.shape[0]-1))
        for i in range(self.robot_bgr_images.shape[0]):
            robot_folder = f"{location}/robot/{str(i).zfill(padding)}"
            os.makedirs(robot_folder, exist_ok=True)

            cv2.imwrite(f"{robot_folder}/rgb.png", self.robot_bgr_images[i])
            np.save(f"{robot_folder}/xyz.npy", self.robot_xyz_images[i])

            with open(f"{robot_folder}/robot_base_t_robot_camera.json", 'w') as f:
                json.dump(self.robot_base_t_robot_camera_s[i].tolist(), f, indent=4)

    def visualize_3d_data(self, visualize:bool = True):
        """
        Visualizes the robot environment using open3d
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.robot_xyz_images.reshape(-1,3))
        pcd.colors = o3d.utility.Vector3dVector(self.robot_bgr_images.reshape(-1,3).astype(np.float32)[:, ::-1]/255)

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

        robot_camera_s = []
        for b_t_c in self.robot_base_t_robot_camera_s:
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            cam_frame.transform(b_t_c)
            robot_camera_s.append(cam_frame)
            robot_camera_s.append(create_3d_camera(
                base_t_camera=b_t_c,
                intrinsics=self.robot_bgr_intrinsics,
                hxw_img=self.robot_bgr_images[0],
                scale=0.1
            ))

        to_vis = [pcd, base_frame]+robot_camera_s
        if visualize:
            o3d.visualization.draw_geometries(to_vis, f"Robot environment: {self.name} visualization")
        return to_vis

    @property
    def name(self)->str:
        return self._name

    @property
    def robot_bgr_images(self)->np.ndarray:
        return self._robot_bgr_images

    @property
    def robot_bgr_intrinsics(self)->np.ndarray:
        return self._robot_bgr_intrinsics

    @property
    def robot_xyz_images(self)->np.ndarray:
        return self._robot_xyz_images

    @property
    def robot_base_t_robot_camera_s(self)->np.ndarray:
        return self._robot_base_t_robot_camera_s