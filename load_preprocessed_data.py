import numpy as np
import cv2
import json
import os
import open3d as o3d

def load_preprocessed_data(load_path:str):
    """
    Loads by stage 1 of the pipeline processed data to feed into the different localization methods
    Expects a folder of the following format:

    `load_path`
    ├── headset
    │   └── headset_image.png from the headset camera
    ├── headset_cam_calibration.json
    ├── robot_cam_calibration.json
    ├── robot
    │   └── filenames (multiple folders)
    │       ├── A xyz.npy file with the world points associated to each pixel
    │       ├── A robot_base_t_robot_camera.json with the 4x4 transformation matrix between robot base and camera
    │       ├── A label.json file with the robot base 2 headset pose estimates based on the aruco marker (optional)
    │       └── A rgb.png taken at the pose defined by `obot_base_t_robot_camera.json`
    └── point_cloud.ply

    The label.json files may not be provided, in that case no ground truth labels can be provided to compare later on

    both calibration.json files must have a field "camera_matrix" which stores a 2D list of the 3x3 camera matrix

    :param load_path: The path from where the data should be loaded
    :return A dictionary with the following keys:
    headset_image: A WxHx3-uint8 RGB image taken by the headset
    headset_cam_matrix: The intrinsic headset camera matrix as a 3x3 numpy array
    robot_cam_matrix: The intrinsic robot camera matrix as a 3x3 numpy array
    robot_datapoints: A list of dictionaries containing the fields: name, rgb_image, xyz, robot_base_t_camera, label (label may be none)
    point_cloud: The point cloud as a Nx3-float numpy array
    """

    headset_image = cv2.cvtColor(cv2.imread(f"{load_path}/headset/headset_image.png"), cv2.COLOR_BGR2RGB)


    headset_cam_calibration = json.load(open(f"{load_path}/headset_cam_calibration.json"))
    robot_cam_calibration = json.load(open(f"{load_path}/robot_cam_calibration.json"))

    robot_positions = []
    for folder in os.listdir(f"{load_path}/robot"):
        rgb_image = cv2.cvtColor(cv2.imread(f"{load_path}/robot/{folder}/rgb.png"), cv2.COLOR_BGR2RGB)
        xyz = np.load(f"{load_path}/robot/{folder}/xyz.npy", allow_pickle=False)
        robot_base_t_camera = np.array(json.load(open(f"{load_path}/robot/{folder}/robot_base_t_robot_camera.json")))

        label = None
        if os.path.exists(f"{load_path}/robot/{folder}/label.json"):
            label = np.array(json.load(open(f"{load_path}/robot/{folder}/label.json")))

        robot_positions.append({
            "name":folder,
            "rgb_image": rgb_image,
            "xyz": xyz,
            "robot_base_t_camera": robot_base_t_camera,
            "label": label,
        })

    point_cloud = o3d.io.read_point_cloud(f"{load_path}/pointcloud.ply")
    point_cloud = np.asanyarray(point_cloud.points)

    return {
        "headset_image": headset_image,
        "headset_cam_matrix": np.array(headset_cam_calibration["camera_matrix"]),
        "robot_cam_matrix": np.array(robot_cam_calibration["camera_matrix"]),
        "robot_datapoints": robot_positions,
        "point_cloud": point_cloud
    }


if __name__ == "__main__":
    print(load_preprocessed_data("out_data")["point_cloud"].shape)
