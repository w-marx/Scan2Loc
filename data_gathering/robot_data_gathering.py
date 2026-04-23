import os
import numpy as np
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys import config_root
import pyrealsense2 as rs
import cv2
import json
import argparse

positions = [
        [-0.36198, -0.049747, 0.033045, -1.6585, 0.20059, 1.6262, 0.35139],
        [-0.199427,-0.303889,0.0692703,-2.43281,0.0363531,1.98839,0.840368],
        [-0.338833,-0.111088,-0.288847,-2.20117,0.307392,1.73006,0.645364],
        [-0.22489,0.532639,-0.192103,-1.35517,0.13621,1.31752,0.0115306],
        [0.277271,0.54268,-0.209407,-1.42414,-0.0531952,1.39126,1.88619],
        [0.317017,-0.139508,0.109578,-2.12047,-0.397975,1.83889,2.40216],
        [0.27664,-0.600108,0.0578287,-2.37475,-0.554525,1.72298,-0.0554051],
        [0.0382363,-0.37407,-0.0369735,-1.61362,-0.139072,1.22564,0.876148],
        [1.04209,-0.82438,-0.866144,-1.86998,-0.341921,1.16218,0.7334],
        [1.34447,-1.05191,-1.01728,-1.80784,-0.622365,1.19451,1.57682],
        [1.42212,-0.691195,-1.4098,-2.11549,-0.692469,1.53,1.58114],
        [0.997678,0.422432,-1.19506,-1.82503,0.218938,1.51332,1.65147],
        [0.513488,-0.0925881,-0.612378,-1.90421,-0.101932,1.45205,1.12216],
        [-0.395858,-0.523988,0.200459,-2.04288,0.0372969,1.48128,0.889949],
        [0.12325, -0.00228, -0.09325, -2.13569, -0.02817, 2.01549, 0.82194]
    ]

DICTIONARY = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
DICTIONARY_ID = 33

def save_intrinsics(rgb_intrinsics, depth_intrinsics, output_folder:str = None, filename:str = None) -> tuple[np.ndarray,list[float],np.ndarray, list[float]]:
    """
    Takes intrinsicy and saves them as a json file if output_folder and filename is not none
    Then returns the intrinsic rgb_camera_mat & coefficients and the intrinsic depth_camera_mat and coefficients
    :param rgb_intrinsics: The intrinsics of a rgb camera
    :param depth_intrinsics: The intrinsics of a depth camera
    :param output_folder: The output folder, may be none if saving is not wished. If it doesnt exist it will be created
    :param filename: The name to save the file under
    :return: he intrinsic rgb_camera_mat & distortion_coefficients & depth_camera_mat and distortion_coefficients
    """
    rgb_camera_mat = np.array([[rgb_intrinsics.fx, 0, rgb_intrinsics.ppx], [0, rgb_intrinsics.fy, rgb_intrinsics.ppy], [0,0,1]])
    depth_camera_mat = np.array([[depth_intrinsics.fx, 0, depth_intrinsics.ppx], [0, depth_intrinsics.fy, depth_intrinsics.ppy], [0,0,1]])

    camera_data = {
        'rgb_camera_matrix': rgb_camera_mat.tolist(),
        'rgb_distortion_coefficients': rgb_intrinsics.coeffs,
        'depth_camera_matrix': depth_camera_mat.tolist(),
        'depth_distortion_coefficients': depth_intrinsics.coeffs,
    }
    if output_folder is not None and filename is not None:
        os.makedirs(f"{output_folder}", exist_ok = True)
        with open(f"{output_folder}/{filename}.json", 'w') as f:
            json.dump(camera_data, f, indent=4)

    return rgb_camera_mat, rgb_intrinsics.coeffs, depth_camera_mat, depth_intrinsics.coeffs

def assemble_homogeneous_matrix(rvec:np.ndarray, tvec:np.ndarray) -> np.ndarray:
    """
    Takes an rotation and translation vector and returns the homogeneous transformation matrix
    :param rvec: rotation vector
    :param tvec: translation vector
    :return: 4x4 numpy array
    """
    transformation = np.eye(4)
    transformation[:3, :3] = cv2.Rodrigues(rvec)[0]
    transformation[:3, 3] = tvec.flatten()
    return transformation



def gather_robot_imgs_eefs(robot_interface, image_pipeline, depth_scale:float, positions:list[list[float]]) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Takes an robot and image interface and moves the robot to the positions.
    It gathers a rgb and depth image at every position and returns them (depth image scaled)+ the endeffector pose as lists
    :param robot_interface:
    :param image_pipeline:
    :param depth_scale:
    :param positions:
    :return: tuple: list of depth images, list of rgb_images, list of gripper to base homogeneous matrices
    """
    depth_images = []
    rgb_images = []
    gripper_t_base_s = []

    for frame_idx, position in enumerate(positions):
        print(f"moving to position: {position}")

        reset_joints_to(robot_interface, position)

        gripper_t_base = robot_interface.last_eef_pose

        frames = image_pipeline.wait_for_frames()
        rgb_frame = np.asanyarray(frames.get_color_frame().get_data())
        depth_frame = np.asanyarray(frames.get_depth_frame().get_data())
        depth_frame_scaled = depth_frame * depth_scale

        depth_images.append(depth_frame_scaled)
        rgb_images.append(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
        gripper_t_base_s.append(gripper_t_base)
    return depth_images, rgb_images, gripper_t_base_s


def estimate_camera_aruco_pose(images: np.ndarray, camera_matrix: np.ndarray, distortion_coefficients: np.ndarray, marker_side_length: float = 0.1) -> list[np.ndarray | None]:
    """
    :param images: NxWxHx3-uint8 RGB images
    :param camera_matrix: 3x3 camera matrix
    :param distortion_coefficients: array of the distortion coefficients of the camera
    :param marker_side_length: side length of the aruco marker in meters
    :return: a list of 4x4 Camera^T_ArucoMarker estimates or None if an image has no aruco marker
    """
    detector_params = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(DICTIONARY, detector_params)

    marker_points = np.array([
        [-marker_side_length / 2, marker_side_length / 2, 0],
        [marker_side_length / 2, marker_side_length / 2, 0],
        [marker_side_length / 2, -marker_side_length / 2, 0],
        [-marker_side_length / 2, -marker_side_length / 2, 0],
    ])

    camera_t_aruco_s = []
    for index, image in enumerate(images):
        marker_corners, marker_ids, reject_candidates = detector.detectMarkers(image)

        if len(marker_corners) > 1:
            raise Exception("More then one aruco marker detected, single pose is not calculatable")
        if len(marker_corners) == 0:
            print(f"no marker found at index {index}")
            camera_t_aruco_s.append(None)
            continue

        _, rvec, tvec = cv2.solvePnP(
            objectPoints=marker_points,
            imagePoints=marker_corners[0][0],
            cameraMatrix=camera_matrix,
            distCoeffs=distortion_coefficients,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        camera_t_aruco_s.append(assemble_homogeneous_matrix(rvec=rvec,tvec=tvec))

    return camera_t_aruco_s

def gather_robot_data(output_folder:str = "data", cam_t_gripper: np.ndarray|None = None, marker_side_length:float = 0.1):
    """
    Creates the following output folder format by moving the robot and taking images:

    `output_folder`
    ├── robot
    │   └── multiple folders with the contents:
    │       ├──  A rgb.png image
    │       ├──  A poses.json file
    │       └──  A depth.npy file
    └── robot_cam_calibration.json

    :param output_folder: the name of the output folder
    :return: nothing
    """
    robot_interface = FrankaInterface(config_root + "/charmander.yml", use_visualizer=False)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    pipeline.start(config)

    rgb_intrinsics = pipeline.get_active_profile().get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_intrinsics = pipeline.get_active_profile().get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()
    rgb_cam_mat, rgb_cam_dist_coef, _, _ = save_intrinsics(rgb_intrinsics=rgb_intrinsics, depth_intrinsics=depth_intrinsics, output_folder=output_folder, filename="robot_cam_calibration")
    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

    depth_images, rgb_images, gripper_t_base_s = gather_robot_imgs_eefs(robot_interface, pipeline, depth_scale, positions)

    camera_t_aruco_s = [None] * len(rgb_images)

    if cam_t_gripper is None:
        print("Using aruco markers for cam_T_gripper determination")
        camera_t_aruco_s = estimate_camera_aruco_pose(np.array(rgb_images), camera_matrix=rgb_cam_mat, distortion_coefficients=np.array(rgb_cam_dist_coef), marker_side_length=marker_side_length)

        r_gripper_t_base = []
        t_gripper_t_base = []
        r_aruco_t_camera = []
        t_aruco_t_camera = []

        for camera_t_aruco, gripper_t_base in zip(camera_t_aruco_s, gripper_t_base_s):
            if camera_t_aruco is not None:
                r_gripper_t_base.append(gripper_t_base[:3, :3])
                t_gripper_t_base.append(gripper_t_base[:3, 3])

                r_aruco_t_camera.append(camera_t_aruco[:3, :3])
                t_aruco_t_camera.append(camera_t_aruco[:3, 3])

        if len(r_gripper_t_base) < 5:
            print(f"Dangerously few aruco marker images: {len(r_gripper_t_base)}")

        r_cam_t_gripper, t_cam_t_gripper = cv2.calibrateHandEye(r_gripper_t_base, t_gripper_t_base, r_aruco_t_camera, t_aruco_t_camera)
        cam_t_gripper = np.concatenate((np.concatenate((r_cam_t_gripper, t_cam_t_gripper), axis=1), [[0, 0, 0, 1]]), axis=0)



    for i, (depth_image, rgb_image, gripper_t_base) in enumerate(zip(depth_images, rgb_images, gripper_t_base_s)):

        # save robot images
        os.makedirs(f"{output_folder}/robot/{i}", exist_ok=True)
        cv2.imwrite(f"{output_folder}/robot/{i}/rgb.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        np.save(f"{output_folder}/robot/{i}/depth.npy", depth_image)

        print(f"gripper_t_base: {gripper_t_base}")
        print(f"cam_t_gripper:{cam_t_gripper}")
        print(f"camera_t_robot_base: {cam_t_gripper @ gripper_t_base}")

        pose_dict = {
            "robot_base_t_camera": np.linalg.inv(cam_t_gripper @ gripper_t_base).tolist(),
            "gripper_t_base": gripper_t_base.tolist(),
            "camera_t_gripper": cam_t_gripper.tolist(),
            # This one may be none, but can be compared for accuracy:
            "robot_base_t_aruco": (np.linalg.inv(cam_t_gripper @ gripper_t_base) @ camera_t_aruco_s[i]).tolist() if camera_t_aruco_s[i] is not None else None
        }
        with open(f"{output_folder}/robot/{i}/poses.json", 'w') as f:
            json.dump(pose_dict, f, indent=4)
    pipeline.stop()
    print(f"finished data gathering")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", type=str, default="data", help="Output Folder Location")
    parser.add_argument("--use-precomputed-cam-t-gripper", action="store_true", default=False, help="If a precomputed cam_t_gripper should be used, if not will be estimated")
    parser.add_argument("--cam-t-gripper-path", type=str, default=None, help="Path to cam_t_gripper.npy")
    parser.add_argument("--aruco-marker-size", type=float, default=0.072, help="Aruco marker size in meters")
    args = parser.parse_args()

    print("loaded cam 2 gripper: ")
    print(np.load("/workspace/franka_pipeline/data/cam_t_gripper.npy"))

    cam_t_gripper = None
    if args.use_precomputed_cam_t_gripper and args.cam_t_gripper_path is not None and os.path.exists(args.cam_t_gripper_path):
        cam_t_gripper = np.load(args.cam_t_gripper_path)
        print(f"Using precomputed cam_t_gripper from {args.cam_t_gripper_path}")

    gather_robot_data(
        output_folder=args.output_folder,
        cam_t_gripper=cam_t_gripper,
        marker_side_length=args.aruco_marker_size

    )

    print("checking results")
    json_files = []
    for folder in os.listdir(f"{args.output_folder}/robot"):
        with open(f"{args.output_folder}/robot/{folder}/poses.json", 'r') as f:
            json_files.append(json.load(f))
    robot_base_t_aruco_s = np.array([pose["robot_base_t_aruco"] for pose in json_files if pose["robot_base_t_aruco"] is not None])
    print(f"recovered {len(robot_base_t_aruco_s)} robot_base_t_aruco's estimates")

    avg_translation = np.mean(np.array([pose[:3, 3] for pose in robot_base_t_aruco_s]), axis=0)
    print(f"average translation: {avg_translation}")

    print("positions:")
    for pose in robot_base_t_aruco_s:
        print(np.round(pose[:3, 3],3))
        print("\n")

    print(f"average translation error: {np.mean(np.linalg.norm(robot_base_t_aruco_s[:,:3,3]-avg_translation, axis=1))}m")

    print("main finished")
