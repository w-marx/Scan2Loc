import cv2
import numpy as np

def assert_intrinsic_mat(m:np.ndarray, hxw_img: np.ndarray | None)->bool:
    assert m.shape == (3, 3), f"M not 3x3 {m.shape}"
    assert m[0, 0] > 0 and m[1, 1] > 0, f"fx and fy must be positive"
    assert np.allclose(m[2, :],[0, 0, 1]), f"bottom row of M incorrect: {m[2:]}"
    assert 0 < m[0, 2] < hxw_img.shape[1] and 0 < m[1, 2] < hxw_img.shape[0], f"center in wrong location {m[0, 2]}, {m[1, 2]} in {hxw_img.shape}"
    return True

def assert_homogeneous_mat(m:np.ndarray, abs_tolerance:float = 0.001) -> bool:
    assert m.shape == (4, 4), f"M not 4x4 {m.shape}"
    assert np.allclose(m[3, :],[0, 0, 0, 1]), f"bottom row of M incorrect: {m[3, :]}"
    assert np.allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=abs_tolerance), f"Rot part not invertible by transpose: {m[:3, :3] @ m[:3, :3].T}"
    assert np.isclose(np.linalg.det(m[:3, :3]), 1, atol=abs_tolerance), f"Determinant is not 1: {np.linalg.det(m[:3, :3])}"
    return True

def create_3d_camera(
        base_t_camera:np.ndarray,
        intrinsics: np.ndarray,
        hxw_img: np.ndarray,
        scale:float = 0.1
    ):
    import open3d as o3d
    assert assert_intrinsic_mat(intrinsics, hxw_img)
    assert assert_homogeneous_mat(base_t_camera)
    assert np.abs(scale) > 1e-6
    fx, fy, cx, cy = intrinsics[0,0], intrinsics[1,1], intrinsics[0,2], intrinsics[1,2]
    w, h = hxw_img.shape[1], hxw_img.shape[0]

    corners_hom = np.array([
        [-cx/fx, -cy/fy, 1.0, 1.0/scale],
        [(w-cx)/fx, -cy/fy, 1.0, 1.0/scale],
        [(w-cx)/fx, (h-cy)/fy, 1.0, 1.0/scale],
        [-cx/fx, (h-cy)/fy, 1.0, 1.0/scale],
        [0,0,0, 1.0/scale]
    ])*scale
    corners = (base_t_camera @ corners_hom.T)[:3, :]
    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(corners.T)
    lines.lines = o3d.utility.Vector2iVector(
        [[0,1], [1,2], [2,3], [3,0], [4,0], [4,1], [4,2], [4,3]]
    )
    return lines


class PredictionData:
    def __init__(
            self,
            name:str,
            robot_bgr_images:np.ndarray,
            robot_bgr_intrinsics:np.ndarray,
            headset_bgr_image: np.ndarray,
            headset_intrinsics: np.ndarray,
            robot_xyz_images:np.ndarray,
            point_cloud:np.ndarray,
            robot_base_t_robot_camera_s:np.ndarray,
            robot_base_t_headset:np.ndarray | None,
    ):
        """
        :param name: the name of the dataset (will be stored under it)
        :param robot_bgr_images: BGR images of the robot as a NxHxWx3-uint8 numpy array
        :param robot_bgr_intrinsics: Intrinsics BGR camera matrix of the robot (3x3 numpy array)
        :param headset_bgr_image: One headset BGR image as a HxWx3-uint8 numpy array
        :param headset_intrinsics: Intrinsics BGR camera matrix of the headset (3x3 numpy array)
        :param robot_xyz_images: XYZ images from the pov of the robot as a NxHxWx3-float numpy array
        :param point_cloud: A point cloud of the surroundings as a Nx3-float numpy array
        :param robot_base_t_robot_camera_s: The homogeneous robot_base->robot_camera transformation matrix as a Nx4x4-float numpy array
        :param robot_base_t_headset: The homogeneous robot_base->robot_headset transformation matrix as a 4x4 matrix or None
        """
        self._name = name

        assert robot_bgr_images.ndim == 4, f"Wrong shape of BGR images {robot_bgr_images.shape}"
        assert robot_bgr_images.shape[0] > 0, f"No BGR images {robot_bgr_images.shape}"
        assert robot_bgr_images.dtype == np.uint8, f"Wrong dtype for bgr images {robot_bgr_images.dtype}"
        self._robot_bgr_images = robot_bgr_images

        assert assert_intrinsic_mat(robot_bgr_intrinsics, robot_bgr_images[0])
        self._robot_bgr_intrinsics = robot_bgr_intrinsics


        assert headset_bgr_image.ndim == 3, f"Wrong shape of Headset image {headset_bgr_image.shape}"
        assert headset_bgr_image.shape[0] > 0 and headset_bgr_image.shape[1] > 0 and  headset_bgr_image.shape[2] == 3, f"No BGR images {robot_bgr_images.shape}"
        assert headset_bgr_image.dtype == np.uint8, f"Wrong dtype for bgr images {headset_bgr_image.dtype}"
        self._headset_bgr_image = headset_bgr_image

        assert assert_intrinsic_mat(headset_intrinsics, headset_bgr_image)
        self._headset_intrinsics = headset_intrinsics


        assert robot_xyz_images.shape == robot_bgr_images.shape
        assert np.issubdtype(robot_xyz_images.dtype, np.floating)
        self._robot_xyz_images = robot_xyz_images

        assert point_cloud.ndim == 2 and point_cloud.shape[0] > 0 and point_cloud.shape[-1] == 3, f"Wrong shape of point cloud {point_cloud.shape}"
        assert np.issubdtype(point_cloud.dtype, np.floating)
        self._point_cloud = point_cloud

        assert robot_base_t_robot_camera_s.shape == (robot_bgr_images.shape[0],4,4), f"Wrong shape of robot_base_t_robot_camera_s {robot_base_t_robot_camera_s.shape}"
        assert all([assert_homogeneous_mat(m) for m in robot_base_t_robot_camera_s])
        self._robot_base_t_robot_camera_s = robot_base_t_robot_camera_s

        assert robot_base_t_headset is None or assert_homogeneous_mat(robot_base_t_headset)
        self._robot_base_t_headset = robot_base_t_headset


    @classmethod
    def from_folder(cls, load_folder:str):
        """
        Loads from a folder of the structure:
        `output_folder`
        ├── headset.png an image with possible aruco markers digitally removed
        ├── headset_cam_calibration.json
        ├── robot_cam_calibration.json
        ├── robot
        │   └── 00 to number of datapoints
        │       ├── A xyz.npy file with the world points associated to each pixel
        │       ├── A robot_base_t_robot_camera.json with the 4x4 transformation matrix between robot base and camera
        │       └── A rgb.png image with possible aruco markers digitally removed
        ├── A label.json file with the robot base -> headset pose truth (might be missing)
        └── point_cloud.ply
        """
        import os, json, open3d

        robot_cam_cal = json.load(open(f"{load_folder}/robot_cam_calibration.json"))
        headset_cam_cal = json.load(open(f"{load_folder}/headset_cam_calibration.json"))
        robot_base_t_headset = np.array(json.load(open(f"{load_folder}/label.json"))) if os.path.exists(f"{load_folder}/label.json") else None

        robot_folders = sorted([f"{folder}" for folder in os.listdir(f"{load_folder}/robot")])
        robot_folders = [f"{load_folder}/robot/{folder}" for folder in robot_folders]

        robot_bgr_images = np.array([cv2.imread(f"{folder}/rgb.png") for folder in robot_folders])
        robot_xyz_images = np.array([np.load(f"{folder}/xyz.npy") for folder in robot_folders])
        robot_base_t_robot_cam_s = np.array([json.load(open(f"{folder}/robot_base_t_robot_camera.json")) for folder in robot_folders])
        point_cloud = np.asarray(open3d.io.read_point_cloud(f"{load_folder}/pointcloud.ply").points)

        instance = cls(
            name = os.path.basename(load_folder),
            robot_bgr_images = robot_bgr_images,
            robot_bgr_intrinsics = np.array(robot_cam_cal["intrinsic_camera_matrix"]),
            headset_bgr_image=cv2.imread(f"{load_folder}/headset.png"),
            headset_intrinsics=np.array(headset_cam_cal["intrinsic_camera_matrix"]),
            robot_xyz_images=robot_xyz_images,
            point_cloud=point_cloud,
            robot_base_t_robot_camera_s=robot_base_t_robot_cam_s,
            robot_base_t_headset= robot_base_t_headset
        )
        return instance

    def save(self, folder_path:str, new_name:str|None=None):
        """
        Saves the instance to a folder, in the format that it can be recreated using `from_folder`.
        Will delete a folder if `folder_path/name` already exists.
        :param folder_path: The folder in which the instance should be saved
        :param new_name: The new name of the output_folder if not they will just use the name
        """
        import os, shutil, json
        import open3d
        location = f"{folder_path}/{new_name if new_name is not None else self.name}"

        if os.path.exists(location):
            print(f"Output folder already exists, deleting it ...")
            shutil.rmtree(location)

        os.makedirs(name=location, exist_ok=True)
        cv2.imwrite(f"{location}/headset.png", self.headset_bgr_image)


        with open(f"{location}/headset_cam_calibration.json", 'w') as f:
                json.dump({'intrinsic_camera_matrix': self.headset_intrinsics.tolist()}, f, indent=4)
        with open(f"{location}/robot_cam_calibration.json", 'w') as f:
            json.dump({'intrinsic_camera_matrix': self.robot_bgr_intrinsics.tolist()}, f, indent=4)

        if self.robot_base_t_headset is not None:
            with open(f"{location}/label.json", 'w') as f:
                json.dump(self.robot_base_t_headset.tolist(), f, indent=4)

        padding = len(str(self.robot_bgr_images.shape[0]-1))
        for i in range(self.robot_bgr_images.shape[0]):
            robot_folder = f"{location}/robot/{str(i).zfill(padding)}"
            os.makedirs(robot_folder, exist_ok=True)

            cv2.imwrite(f"{robot_folder}/rgb.png", self.robot_bgr_images[i])
            np.save(f"{robot_folder}/xyz.npy", self.robot_xyz_images[i])

            with open(f"{robot_folder}/robot_base_t_robot_camera.json", 'w') as f:
                json.dump(self.robot_base_t_robot_camera_s[i].tolist(), f, indent=4)

        point_cloud_o3d = open3d.geometry.PointCloud()
        point_cloud_o3d.points = open3d.utility.Vector3dVector(self.point_cloud)
        open3d.io.write_point_cloud(f"{location}/pointcloud.ply", point_cloud_o3d, write_ascii=True)

    def visualize_3d_data(self):
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.point_cloud)

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

        robot_camera_s = []
        for idx, b_t_c in enumerate(self.robot_base_t_robot_camera_s):
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

        if self.robot_base_t_headset is not None:
            headset_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
            to_vis.append(create_3d_camera(
                base_t_camera=self.robot_base_t_headset,
                intrinsics=self.headset_intrinsics,
                hxw_img=self.headset_bgr_image,
                scale=0.2
            ))
            headset_frame.transform(self.robot_base_t_headset)
            to_vis.append(headset_frame)

        o3d.visualization.draw_geometries(to_vis, f"Processed Data {self.name} visualization")

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
    def headset_bgr_image(self)->np.ndarray:
        return self._headset_bgr_image

    @property
    def headset_intrinsics(self)->np.ndarray:
        return self._headset_intrinsics

    @property
    def robot_xyz_images(self)->np.ndarray:
        return self._robot_xyz_images

    @property
    def point_cloud(self)->np.ndarray:
        return self._point_cloud

    @property
    def robot_base_t_robot_camera_s(self)->np.ndarray:
        return self._robot_base_t_robot_camera_s

    @property
    def robot_base_t_headset(self)->np.ndarray | None:
        return self._robot_base_t_headset