import json, os, shutil, logging
import open3d as o3d
import cv2
import numpy as np

from shared.se3_utilities import compute_pose_pseudo_median
from shared.image_camera_manipulation import create_3d_camera, crop_images, scale_images
from shared.assertion_helpers import assert_mxnx3_np_uint8_image_batch, assert_intrinsic_mat, assert_homogeneous_mat
from shared.gathered_robot_data import GatheredRobotData


class HeadsetData:
    def __init__(self,
                 name:str,
                 bgr_image_s: np.ndarray,
                 intrinsic_cam_mtx: np.ndarray,
                 robot_base_t_headset_s: list[np.ndarray | None]
                 ):
        """
        :param name: the name of the dataset (will be stored under it)
        :param bgr_image_s: The headset BGR images as a NxHxWx3-uint8 numpy array
        :param intrinsic_cam_mtx: Intrinsics BGR camera matrix of the headset (3x3 numpy array)
        :param robot_base_t_headset_s: a list of 4x4 robot_base T_headsets hom. transformations or None
        """
        self._n_frames = bgr_image_s.shape[0]
        assert self._n_frames > 0, f"Number of images must be greater then 0: {bgr_image_s}"

        self._name = name

        assert assert_mxnx3_np_uint8_image_batch(bgr_image_s)
        self._bgr_image_s = bgr_image_s

        assert assert_intrinsic_mat(intrinsic_cam_mtx, bgr_image_s[0])
        self._intrinsic_cam_mtx = intrinsic_cam_mtx

        assert len(robot_base_t_headset_s) == self._n_frames, f"Label/image dimension mismatch: {self._n_frames} != {len(robot_base_t_headset_s)}"
        assert all(m is None or assert_homogeneous_mat(m, size=4) for m in robot_base_t_headset_s)
        self._robot_base_t_headset_s = robot_base_t_headset_s
    
    @classmethod
    def from_folder(cls, load_folder:str)->'HeadsetData':
        """
        Loads from a folder of the structure:
        `load_folder`
        ├── headset_cam_calibration.json
        └── headset
            └── 00 to number of datapoints
                ├── A label.json with the robot base -> headset pose truth (might be missing)
                └── A rgb.png image with possible aruco markers digitally removed
        """

        headset_cam_cal = json.load(open(f"{load_folder}/headset_cam_calibration.json"))

        headset_folders = sorted([f"{folder}" for folder in os.listdir(f"{load_folder}/headset")])
        headset_folders = [f"{load_folder}/headset/{folder}" for folder in headset_folders]
        headset_bgr_image_s = np.array([cv2.imread(f"{folder}/rgb.png") for folder in headset_folders])

        robot_base_t_headset_s = []
        for folder in headset_folders:
            label_path = f"{folder}/label.json"
            if os.path.exists(label_path):
                robot_base_t_headset_s.append(np.array(json.load(open(label_path))))
            else:
                robot_base_t_headset_s.append(None)

        instance = cls(
            name = os.path.basename(load_folder),
            intrinsic_cam_mtx=np.array(headset_cam_cal["intrinsic_camera_matrix"]),
            bgr_image_s= headset_bgr_image_s,
            robot_base_t_headset_s = robot_base_t_headset_s
        )
        return instance
    
    @classmethod
    def from_vrs_file(cls, file_location:str)->'HeadsetData':
        """
        This function takes a .vrs file and creates a HeadsetData instance
        It undistorts the images by taking the camera-rgb - fisheye camera and transforming it to a pinhole camera
        It makes the assumption that the fisheye camera focal length is the average of fx and fy.
        The distortion coefficients are assumed to be 0

        :param file_location: The location of the .vrs file including the filename
        :return: an instance of Headset Data
        """
        from projectaria_tools.core import data_provider, calibration
        if not os.path.isfile(file_location):
            raise FileNotFoundError(f"No .vrs file found at {file_location}")

        provider = data_provider.create_vrs_data_provider(file_location)
        stream_id = provider.get_stream_id_from_label("camera-rgb")
        cam_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
        img_width, img_height = cam_calib.get_image_size()
        focal_length = (cam_calib.get_focal_lengths()[0] + cam_calib.get_focal_lengths()[1]) / 2
        pinhole = calibration.get_linear_camera_calibration(image_width=img_width, image_height=img_height,focal_length=focal_length, label="camera-rgb")

        bgr_hxw_imgs = []
        for i in range(0, provider.get_num_data(stream_id)):
            image_data = provider.get_image_data_by_index(stream_id, i)[0].to_numpy_array()
            undistorted_image = calibration.distort_by_calibration(arraySrc=image_data, dstCalib=pinhole,srcCalib=cam_calib)
            bgr_image = cv2.cvtColor(np.array(undistorted_image), cv2.COLOR_RGB2BGR)
            bgr_hxw_imgs.append(cv2.rotate(bgr_image, cv2.ROTATE_90_CLOCKWISE))

        fx, fy = pinhole.get_focal_lengths()
        cx, cy = pinhole.get_principal_point()
        mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

        return cls(
            name = os.path.splitext(os.path.basename(file_location))[0],
            bgr_image_s= np.array(bgr_hxw_imgs),
            intrinsic_cam_mtx= mtx,
            robot_base_t_headset_s = [None] * len(bgr_hxw_imgs)
        )
    
        
    def resized_and_cropped(self, *, crop_amount: tuple[int, int] | None = None, new_res: tuple[int, int] | None = None)->'HeadsetData':
        """
        Returns a copy of the instance with the new resolution
        :param crop_amount: The amount to be cropped before scaling to the new resolution (crop_w, crop_h) (applied twice on each side)
        :param new_res: The new resolution as an (widht, height) tuple
        :return: A copy where the images have the new resolution
        """
        new_images, new_intrinsics = self.bgr_image_s.copy(), self.intrinsic_cam_mtx.copy()
        if crop_amount is not None:
            new_images, new_intrinsics = crop_images(new_images, new_intrinsics, crop_amount)
        if new_res is not None:
            new_images, new_intrinsics = scale_images(new_images, new_intrinsics, new_res)

        return HeadsetData(
            name=self.name,
            bgr_image_s=new_images,
            intrinsic_cam_mtx=new_intrinsics,
            robot_base_t_headset_s=self.robot_base_t_headset_s.copy()
        )

    
    def save(self, folder_path:str, new_name:str|None=None):
        """
        Saves the instance to a folder, in the format that it can be recreated using `from_folder`.
        Will delete a folder if `folder_path/name` already exists.
        :param folder_path: The folder in which the instance should be saved
        :param new_name: The new name of the output_folder if not they will just use the name
        """
        location = f"{folder_path}/{new_name if new_name is not None else self.name}"

        if os.path.exists(location):
            logging.info(f"Output folder already exists, deleting it ...")
            shutil.rmtree(location)

        os.makedirs(name=location, exist_ok=True)

        with open(f"{location}/headset_cam_calibration.json", 'w') as f:
                json.dump({'intrinsic_camera_matrix': self.intrinsic_cam_mtx.tolist()}, f, indent=4)

        padding = len(str(self._n_frames-1))
        for i in range(self._n_frames):
            headset_folder = f"{location}/headset/{str(i).zfill(padding)}"
            os.makedirs(headset_folder, exist_ok=True)

            cv2.imwrite(f"{headset_folder}/rgb.png", self.bgr_image_s[i])

            if self.robot_base_t_headset_s[i] is not None:
                with open(f"{headset_folder}/label.json", 'w') as f:
                    json.dump(self.robot_base_t_headset_s[i].tolist(), f, indent=4)

    def visualize_3d_data(self, visualize:bool = True):
        """
        Visualizes the headset trajectory using open3d
        :param visualize: If true, visualizes the camera poses
        :return: The list of open3d geometries that were visualized
        """
        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

        to_vis = [base_frame]
        for b_t_c in self.robot_base_t_headset_s:
            if b_t_c is not None:
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
                cam_frame.transform(b_t_c)
                to_vis.append(cam_frame)
                to_vis.append(create_3d_camera(
                    base_t_camera=b_t_c,
                    intrinsics=self.intrinsic_cam_mtx,
                    hxw_img=self.bgr_image_s[0],
                    scale=0.1
                ))
        if visualize:
            o3d.visualization.draw_geometries(to_vis, f"Headset Recording: {self.name} visualization")
        return to_vis


    @property
    def name(self)->str:
        return self._name

    @property
    def bgr_image_s(self)->np.ndarray:
        return self._bgr_image_s

    @property
    def intrinsic_cam_mtx(self)->np.ndarray:
        return self._intrinsic_cam_mtx
    
    @property
    def robot_base_t_headset_s(self)->list[np.ndarray | None]:
        return self._robot_base_t_headset_s

    @property
    def labeled_frames_indices(self)->list[int]:
        return [i for i, m in enumerate(self.robot_base_t_headset_s) if m is not None]

    @property
    def n_frames(self):
        return self._n_frames


def create_robot_bound_headset_data(
        headset_data:HeadsetData,
        robot_data:GatheredRobotData,
    )->HeadsetData:
    """
    Uses the robot_data to add labels to an HeadsetData object
    :param headset_data: A HeadsetData instance that will be the blueprint for the new object
    :param robot_data: A GatheredRobotData instance, that provides the Robot->Marker transformations and marker detector
    :return: An HeadsetData object (can be the same as the input object)
    """

    robot_base_t_marker_s = [m for m in robot_data.base_t_marker_s if m is not None]

    if robot_data.marker_detector is None:
        return headset_data

    headset_t_markers = robot_data.marker_detector.get_camera_t_marker(
        images=list(headset_data.bgr_image_s),
        camera_matrix=headset_data.intrinsic_cam_mtx,
    )

    headset_t_markers_no_none_idx = [i for i, m in enumerate(headset_t_markers) if m is not None]
    if len(robot_base_t_marker_s) == 0 or len(headset_t_markers_no_none_idx) == 0:
        return headset_data
    
    robot_base_t_marker = compute_pose_pseudo_median(robot_base_t_marker_s)

    base_t_headsets = [
        ((robot_base_t_marker @ np.linalg.inv(h_t_m)) if (h_t_m is not None and robot_base_t_marker is not None) else None)
        for h_t_m in headset_t_markers
    ]

    masked_headset_images = robot_data.marker_detector.remove_markers(list(headset_data.bgr_image_s))

    return HeadsetData(
        name=headset_data.name,
        bgr_image_s=np.array(masked_headset_images),
        intrinsic_cam_mtx=headset_data.intrinsic_cam_mtx,
        robot_base_t_headset_s=base_t_headsets
    )