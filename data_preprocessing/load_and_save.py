import sys
import os
import json
import shutil
import open3d as o3d
from projectaria_tools.core import data_provider, calibration

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gathering.aruco_charuco_detection import *

def vrs_to_images_intrinsic(file_location:str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """
    Converts the rgb channel of the file at the location to an array of images (undistorted) and returns the intrinsic camera matrix.
    It undistorts them by taking the camera-rgb - fisheye camera and transforming it to a pinhole camera
    It makes the assumption that the fisheye camera focal length is the average of fx and fy.
    The distortion coefficients are assumed to be 0
    :param file_location: The location of the .vrs file including the filename
    :return: an NxWxHx3-uint8 RGB image array, the camera intrinsic matrix, the camera distortion coefficients
    """
    if not os.path.isfile(file_location):
        raise FileNotFoundError(f"No .vrs file found at {file_location}")

    provider = data_provider.create_vrs_data_provider(file_location)
    stream_id = provider.get_stream_id_from_label("camera-rgb")
    cam_calib = provider.get_device_calibration().get_camera_calib("camera-rgb")
    img_width, img_height = cam_calib.get_image_size()
    focal_length = (cam_calib.get_focal_lengths()[0]+cam_calib.get_focal_lengths()[1])/2
    pinhole = calibration.get_linear_camera_calibration(image_width=img_width, image_height=img_height, focal_length=focal_length, label="camera-rgb")

    images = []
    for i in range(0, provider.get_num_data(stream_id)):
        image_data = provider.get_image_data_by_index(stream_id, i)[0].to_numpy_array()
        undistorted_image = calibration.distort_by_calibration(arraySrc=image_data, dstCalib=pinhole, srcCalib=cam_calib)
        images.append(undistorted_image)

    fx, fy = pinhole.get_focal_lengths()
    cx, cy = pinhole.get_principal_point()
    mtx = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return np.array(images), mtx, [0,0,0,0,0]




def load_input_data(input_folder:str) -> dict[str, np.ndarray | None | list[str] | list[float] | list[np.ndarray]]:
    """
    Takes the location of a input data folder and loads it into memory
    Input data folder should have the following structure:

    `input_folder`
    ├── headset.vrs
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

    :param input_folder: the location of the input data folder
    :return: A Dictionary:
    {
        headset_images: headset_images (NxWxHx3-uint8 numpy array),
        headset_cam_mtx: 3x3 numpy array of the cam_mtx or None (either headset_calibration_images or robot_cam_mtx is not none)
        headset_list_coefficients: A list of floats and with the headset cam distortion coefficients

        robot_rgb_images:  (NxWxHx3-uint8 numpy array),
        robot_depth_images: (NxWxH numpy array)
        robot_images_names: robot_image_names list of strings of length number of robot images,
        robot_rgb_cam_mtx: A 3x3 Numpy array with the intrinsic matrix of the rgb camera
        robot_depth_cam_mtx: A 3x3 Numpy array with the intrinsic matrix of the depth camera
        robot_depth_dist: A list of floats and with the robot rgb cam distortion coefficients
        robot_rgb_dist: A list of floats and with the robot_rgb cam distortion coefficients

        robot_base_t_camera_s: A list of Nx4x4 numpy of transformation matrices or None if those are not provided
        robot_camera_t_marker_s: A list of Nx4x4 numpy of transformation matrices or None if those are not provided
        aruco_marker_detector: A ArucoMarkerDetector object or None if the aruco marker detector was not provided
    }
    """

    # vrs file handling
    vrs_files = [file for file in os.listdir(f"{input_folder}") if file.endswith('.vrs')]
    if len(vrs_files) == 0:
        raise Exception(f"No VRS files found at {input_folder}", FileNotFoundError)
    if len(vrs_files) > 1:
        print(f"Found multiple .vrs files, using {vrs_files[0]}")
    headset_images, headset_mtx, headset_dist_coef = vrs_to_images_intrinsic(f"{input_folder}/{vrs_files[0]}")

    # Load Robot images
    robot_folder_names = [f"{folder_name}" for folder_name in os.listdir(f"{input_folder}/robot")]
    try:
        robot_folder_names = sorted(robot_folder_names, key=lambda x: int(x))
    except Exception:
        print("could not sort robot images as numbers")


    robot_rgb_images, robot_depth_images, robot_base_t_cameras, robot_camera_t_marker = [], [], [], []

    for folder in robot_folder_names:
        location = f"{input_folder}/robot/{folder}"

        robot_rgb_images.append(cv2.cvtColor(cv2.imread(f"{location}/rgb.png"), cv2.COLOR_BGR2RGB))
        robot_depth_images.append(np.load(f"{location}/depth.npy"))

        poses_dict = json.load(open(f"{location}/poses.json"))
        robot_base_t_cameras.append(np.array(poses_dict["base_t_cam"]))
        robot_camera_t_marker.append(np.array(poses_dict["camera_t_marker"]) if poses_dict["camera_t_marker"] is not None else None)


    # Load Robot calibration
    robot_calibration = json.loads(open(f"{input_folder}/robot_cam_calibration.json").read())

    # Load aruco marker
    aruco_marker_detector = None
    if os.path.exists(f"{input_folder}/metadata.json"):
        aruco_marker_detector = build_aruco_charuco_detector(json.load(open(f"{input_folder}/metadata.json")))

    ret_dict = {
        "headset_images": headset_images,
        "headset_cam_mtx": headset_mtx,
        "headset_dist_coef": headset_dist_coef,

        "robot_rgb_images": robot_rgb_images,
        "robot_depth_images": robot_depth_images,
        "robot_folder_names": robot_folder_names,
        "robot_base_t_camera_s": robot_base_t_cameras,
        "robot_camera_t_marker_s": robot_camera_t_marker,

        "robot_rgb_cam_mtx": np.array(robot_calibration["rgb_camera_matrix"]),
        "robot_depth_cam_mtx": np.array(robot_calibration["depth_camera_matrix"]),
        "robot_depth_dist": robot_calibration["depth_distortion_coefficients"],
        "robot_rgb_dist": robot_calibration["rgb_distortion_coefficients"],
        "marker_detector": aruco_marker_detector
    }

    return ret_dict


def save_cam_properties_as_json(output_folder:str, output_file_name:str, cam_mtx:np.ndarray,cam_dist_coeffs:list[float]):
    """
    Saves the camera properties under the keys: `intrinsic_camera_matrix` and `distortion_coefficients` into a json file
    :param output_folder: The folder to save the json file into
    :param output_file_name: The name of the json file
    :param cam_mtx: A 3x3 camera Matrix
    :param cam_dist_coeffs: An array of distortion coefficients
    :return: Nothing
    """
    assert cam_mtx.shape == (3, 3)
    headset_camera_data = {
        'intrinsic_camera_matrix': cam_mtx.tolist(),
        'distortion_coefficients': cam_dist_coeffs
    }
    if output_folder is not None and output_file_name is not None:
        with open(f"{output_folder}/{output_file_name}.json", 'w') as f:
            json.dump(headset_camera_data, f, indent=4)


def save_output_data(
        output_folder:str,
        headset_image:np.ndarray,
        headset_cam_mtx:np.ndarray,
        headset_cam_dist_coeffs:list[float],
        robot_folder_names: list[str],
        robot_rgb_images:np.ndarray,
        robot_xyz_images:np.ndarray,
        robot_base_t_robot_camera_s:list[np.ndarray],
        robot_base_t_headsets:list[np.ndarray | None],
        robot_rgb_cam_mtx:np.ndarray,
        robot_rgb_cam_dist_coeffs:list[float],
        point_cloud:np.ndarray,
):
    """
    Outputs a Folder of the structure:
    `output_folder`
    ├── headset
    │   └── a .png image with possible aruco markers digitally removed
    ├── headset_cam_calibration.json
    ├── robot_cam_calibration.json
    ├── robot
    │   └── filenames (multiple folders)
    │       ├── A xyz.npy file with the world points associated to each pixel
    │       ├── A robot_base_t_robot_camera.json with the 4x4 transformation matrix between robot base and camera
    │       ├── A label.json file with the robot base 2 headset pose estimates based on the aruco marker (might be missing)
    │       └── A rgb.png image with possible aruco markers digitally removed
    └── point_cloud.ply
    """
    assert headset_image.ndim == 3 and headset_image.shape[-1] == 3
    assert headset_cam_mtx.shape == (3, 3)
    n_datapoints = len(robot_folder_names)
    assert n_datapoints == robot_rgb_images.shape[0] and robot_rgb_images.shape[-1] == 3
    assert n_datapoints == robot_xyz_images.shape[0] and robot_xyz_images.shape == robot_rgb_images.shape, f"n datapoints:{n_datapoints}, xyzimg: {robot_xyz_images.shape}, rgbimg: {robot_rgb_images.shape}"
    assert n_datapoints == len(robot_base_t_robot_camera_s), f"n datapoints: {n_datapoints}, number b_t_c poses: {len(robot_base_t_robot_camera_s)}"
    assert n_datapoints == len(robot_base_t_headsets), f"n datapoints: {n_datapoints}, number b_t_h poses: {len(robot_base_t_robot_camera_s)}"
    assert robot_rgb_cam_mtx.shape == (3, 3), f"robot cam mtx shape: {robot_rgb_cam_mtx.shape}"
    assert point_cloud.ndim == 2 and point_cloud.shape[-1] == 3

    if os.path.exists(f"{output_folder}"):
        print(f"Output folder already exists, deleting it ...")
        shutil.rmtree(f"{output_folder}")

    os.makedirs(name = f"{output_folder}/headset", exist_ok=True)
    cv2.imwrite(f"{output_folder}/headset/headset_image.png", cv2.cvtColor(headset_image, cv2.COLOR_BGR2RGB))

    save_cam_properties_as_json(output_folder, "headset_cam_calibration", headset_cam_mtx, headset_cam_dist_coeffs)
    save_cam_properties_as_json(output_folder, "robot_cam_calibration", robot_rgb_cam_mtx, robot_rgb_cam_dist_coeffs)


    for rgb_image, xyz_image, robot_base_t_robot_camera, robot_base_t_headset, name in zip(robot_rgb_images, robot_xyz_images, robot_base_t_robot_camera_s, robot_base_t_headsets,robot_folder_names):
        os.makedirs(f"{output_folder}/robot/{name}", exist_ok=True)
        cv2.imwrite(f"{output_folder}/robot/{name}/rgb.png", cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        np.save(f"{output_folder}/robot/{name}/xyz.npy", xyz_image)

        with open(f"{output_folder}/robot/{name}/robot_base_t_robot_camera.json", 'w') as f:
            json.dump({"robot_base_t_robot_camera":robot_base_t_robot_camera.tolist()}, f, indent=4)
        with open(f"{output_folder}/robot/{name}/label.json", 'w') as f:
            json.dump({"robot_base_t_headset":robot_base_t_headset.tolist() if robot_base_t_headset is not None else None}, f, indent=4)

    point_cloud_o3d = o3d.geometry.PointCloud()
    point_cloud_o3d.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.io.write_point_cloud(f"{output_folder}/pointcloud.ply", point_cloud_o3d, write_ascii=False)





