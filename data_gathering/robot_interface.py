# robot specific imports
import pyrealsense2
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys import config_root

import pyrealsense2 as rs
import cv2
import numpy as np
import json
import os
import time

# Some positions for the Franka Panda robot.
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
    [-0.05111951, -0.20194183,  0.1768012 , -1.9058693 , -0.10422819,  1.70261123,  1.02698558],
    [-0.00800726, -0.20028432,  0.19629781, -1.71570337, -0.22150046,  1.30892323,  1.02652492],
    [-0.01877233, -0.7274169 , -0.37034855, -2.17385817, -0.04688886,  1.58002037,  0.37590141],
    [ 0.09368807, -0.5680825 , -0.23207449, -2.33412267, -0.10451632,  1.66822976, -0.20135044],
    [ 0.19388837,  0.21229561, -0.14619505, -1.53436363, -0.12444775,  1.36505886,  0.29731757],
    [ 0.03414263,  0.39142182, -0.10106023, -1.54919985, -0.13722469,  1.64687527, -0.48019857],
    [-0.34205451,  0.47926156, -0.11499073, -0.83376411,  0.09258667,  1.00671863, -0.11133599],
    [-0.11337702, -0.50130846, -0.05314591, -2.42742063,  0.02944801,  1.88180914,  0.45898891],
    [-0.16930966, -0.72734304, -0.08863141, -2.8781092 ,  0.03405676,  2.26517989,  0.452756  ],
    [-0.22131274, -0.46883255, -1.013539  , -2.45203124,  0.49748234,  2.07431517, -1.31043617],
    [-0.58812327, -0.64133459,  0.85269879, -2.49271683,  0.22187203,  1.96053645,  1.04365787],
    [-0.53472265,  0.01312082,  0.37475239, -1.01873491,  0.18205305,  1.05653433,  0.89998009],
    [-0.54976133,  0.06693389,  0.3411553 , -1.00700466,  0.18275264,  0.96961443,  1.4328389 ],
    [ 0.56432717,  0.09287738,  0.37899787, -1.0149873 , -0.31956106,  0.91674459,  1.65892086],
    [ 0.62061286,  0.2869147 ,  0.37213654, -1.2337113 , -0.4605359 ,  1.12766345,  2.32680865],
    [ 0.54109234,  0.48444566,  0.38081182, -1.22580852, -0.4918089 ,  1.11899586,  2.46844307],
    [ 0.07408837,  0.01258195,  0.09635632, -1.81522833, -0.06211733,  1.77488479,  2.45850854],
    [-0.14620351, -0.01600424,  0.09240906, -1.85344749, -0.06266853,  1.77423007, -0.8207692 ],
    [-0.11552985, -0.16852483,  0.09648213, -1.92001261, -0.06095093,  1.53294177,  0.70070849],
    [-0.22103787,  0.06906212,  0.09034234, -1.95686574, -0.05603532,  1.58011393,  1.649505  ],
    [ 0.42137264,  0.17752595,  0.35647084, -1.64596009, -0.59505662,  1.17605876,  1.34227659],
    [ 0.17892749, -0.11568239,  0.34051132, -1.77237959, -0.27739304,  1.34717743,  1.35104485],
    [ 0.71856866, -0.13966426, -0.65642883, -1.96710477, -0.19670552,  1.54517216,  1.12532561],
    [ 0.64118936, -0.16718129, -0.71046499, -1.89893176, -0.1476313 ,  1.54290738,  0.88084655],
    [ 0.6290293 , -0.1589057 , -0.6444431 , -1.84843379, -0.15438134,  1.4851713 ,  0.82488483],
    [ 1.12725909, -0.24749665, -1.23004939, -1.90112157, -0.16122164,  1.61067107,  0.7883    ],
    [ 0.94540554,  0.18229158, -1.24794147, -1.74144459,  0.03165758,  1.39620755,  0.94522439],
    [ 1.00783434, -0.01161147, -1.20084634, -1.93957709,  0.02203243,  1.60691821,  0.96291597],
    [ 1.01571717,  0.01804357, -1.1919804 , -1.91762172,  0.02171737,  1.63513622,  0.64757605],
    [ 0.98515699,  0.50775101, -1.10373793, -1.56497076,  0.09496894,  1.4518602 ,  0.64996145],
    [ 1.08895942,  0.35593273, -1.11322819, -1.46121051,  0.05474147,  1.25588019,  0.65330947],
    [ 0.9697643 , -0.40518597, -0.91968736, -2.22278022, -0.35060279,  1.84287091,  0.98557138],
    [ 0.95404639, -0.25632717, -1.1766304 , -2.06135871, -0.32004115,  1.74226022,  0.63039024],
    [ 0.97677385, -0.40183406, -0.97059404, -2.25833686, -0.45449668,  1.92664382,  1.26239766],
    [ 1.16580208,  0.27004659, -0.51991596, -1.90772556, -0.72563669,  1.61837002,  1.259613  ],
    [ 0.54956908,  0.71861543, -0.73861857, -0.94352113,  0.28036504,  1.01700386,  0.77550107],
    [ 0.28256933,  0.72907666, -0.74134145, -0.94156915,  0.43567126,  1.09704004,  0.77550126],
    [ 0.12667911,  0.77368228, -0.74571669, -0.94008991,  0.43378638,  0.95134418,  0.78013267],
    [-0.07971441,  0.83068791, -0.74761757, -0.94322867,  0.53804526,  0.98845405,  0.78337462],
    [-0.34477032,  0.73351253, -0.83461123, -0.94122882,  0.74159557,  0.90435568,  1.06115485],
    [ 1.33406023,  1.74769682, -1.88388773, -1.37111848,  0.98185787,  1.20446507,  2.39218888],
    [ 1.04203811,  1.75249143, -1.99975423, -0.99963038,  1.21306021,  1.04894061,  2.71778564],
    [ 0.9261044 ,  1.75302339, -2.00780353, -1.06652834,  1.42349458,  1.00881587,  2.78927818],
    [ 0.91496365,  1.60199434, -2.05182506, -1.28060508,  1.43350809,  0.8557538 ,  2.7608788 ],
    [ 1.8203415 ,  0.46194938, -2.63320382, -0.96477759,  0.45960628,  0.63502695,  0.36945578],
    [ 1.76705995,  0.23457285, -2.8520838 , -1.0909179 ,  0.53212887,  0.75837838,  0.36877098],
    [ 1.68429794,  0.25609402, -2.84697426, -1.45276892,  0.51931834,  0.96532685,  0.37011954],
    [0.12325, -0.00228, -0.09325, -2.13569, -0.02817, 2.01549, 0.82194],
]

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

def gather_robot_imgs_eefs(
        robot_interface,
        image_pipeline:pyrealsense2.pipeline,
        depth_scale:float,
        robot_positions:list[list[float]],
        number_of_positions:None|int = None,
        stabilisation_timeout:float = 0.0
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Takes an robot and image interface and moves the robot to the positions.
    It gathers a rgb and depth image at every position and returns them (depth image scaled)+ the end effector pose as lists
    :param stabilisation_timeout:
    :param robot_interface:
    :param image_pipeline: A pyrealsense2 pipeline object for the camera attached to the robot end effector
    :param depth_scale: A factor to multiply the depth image by to get the depth in meters
    :param number_of_positions: The number of positions to gather images for. If None it will gather images for all passed positions
    :param robot_positions: A list of positions in joint coordinates (7 floats)
    :return: tuple: list of depth images, list of rgb_images, list of base-to-gripper homogeneous matrices
    """
    depth_images = []
    rgb_images = []
    base_t_gripper_s = []

    for frame_idx, position in enumerate(robot_positions[:number_of_positions if number_of_positions is not None else len(robot_positions)]):
        print(f"moving to position {frame_idx} : {position}")

        reset_joints_to(robot_interface, position)
        time.sleep(stabilisation_timeout)

        base_t_gripper = robot_interface.last_eef_pose

        frames = image_pipeline.wait_for_frames()
        rgb_frame = np.asanyarray(frames.get_color_frame().get_data())
        depth_frame = np.asanyarray(frames.get_depth_frame().get_data())
        depth_frame_scaled = depth_frame * depth_scale
        print(f"depth_frame: {depth_frame_scaled.shape}")
        depth_images.append(depth_frame_scaled)
        rgb_images.append(cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB))
        base_t_gripper_s.append(base_t_gripper)
    return depth_images, rgb_images, base_t_gripper_s


def gather_robot_data(
        output_folder: str = "data",
        number_of_positions:None|int = None,
        stabilisation_timeout:float = 0.0
    ) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray, list[float]]:
    """
    Creates the following output folder format by moving the robot and taking images:

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


    :param number_of_positions: The number of positions to gather images for. If None it will gather images for all passed positions
    :param stabilisation_timeout: the time to wait after moving the robot before taking an image in seconds
    :param output_folder: the name of the output folder

    :return: rgb_images, base_t_gripper_s, rgb_cam_mat, rgb_cam_dist_coef
    """
    robot_interface = FrankaInterface(config_root + "/charmander.yml", use_visualizer=False)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    pipeline.start(config)

    rgb_intrinsics = pipeline.get_active_profile().get_stream(
        rs.stream.color).as_video_stream_profile().get_intrinsics()
    depth_intrinsics = pipeline.get_active_profile().get_stream(
        rs.stream.depth).as_video_stream_profile().get_intrinsics()
    rgb_cam_mat, rgb_cam_dist_coef, _, _ = save_intrinsics(rgb_intrinsics=rgb_intrinsics,
                                                           depth_intrinsics=depth_intrinsics,
                                                           output_folder=output_folder,
                                                           filename="robot_cam_calibration")

    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

    depth_images, rgb_images, base_t_gripper_s = gather_robot_imgs_eefs(robot_interface=robot_interface,
                                                                        image_pipeline=pipeline,
                                                                        depth_scale=depth_scale,
                                                                        robot_positions=positions,
                                                                        number_of_positions=number_of_positions,
                                                                        stabilisation_timeout=stabilisation_timeout)

    for i, (depth_image, rgb_image, base_t_gripper) in enumerate(zip(depth_images, rgb_images, base_t_gripper_s)):
        # save robot images
        # TODO depth images are not saved (are empty)
        os.makedirs(f"{output_folder}/robot/{i:06d}", exist_ok=True)
        cv2.imwrite(f"{output_folder}/robot/{i:06d}/rgb.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
        np.save(f"{output_folder}/robot/{i:06d}/depth.npy", depth_image)

        pose_dict = {
            "base_t_gripper": base_t_gripper.tolist(),
        }
        with open(f"{output_folder}/robot/{i:06d}/poses.json", 'w') as f:
            json.dump(pose_dict, f, indent=4)
    pipeline.stop()
    return rgb_images, base_t_gripper_s, rgb_cam_mat, rgb_cam_dist_coef
