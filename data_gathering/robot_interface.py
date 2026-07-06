# robot specific imports
from deoxys.franka_interface import FrankaInterface
from deoxys.experimental.motion_utils import reset_joints_to
from deoxys import config_root

import pyrealsense2 as rs
import numpy as np
import os, time, logging

from shared.raw_robot_scan import *

from .depth_filtering import *


def extract_intrinsics(cam_intrinsics) -> tuple[np.ndarray,list[float],np.ndarray, list[float]]:
    """
    :param cam_intrinsics: The intrinsics of a rgb camera
    :param depth_intrinsics: The intrinsics of a depth camera
    :return: The cam_intrinsics & distortion_coefficients
    """
    camera_mat = np.array([[cam_intrinsics.fx, 0, cam_intrinsics.ppx], [0, cam_intrinsics.fy, cam_intrinsics.ppy], [0,0,1]])
    return camera_mat, cam_intrinsics.coeffs


def gather_robot_imgs_eefs(
        robot_interface,
        image_pipeline:rs.pipeline,
        depth_scale:float,
        robot_positions:np.ndarray,
        depth_filter_pipeline,
        stabilisation_timeout:float = 0.0,
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
    :return: tuple: list of depth images, list of bgr, list of base-to-gripper homogeneous matrices
    """
    assert stabilisation_timeout >= 0, f"Timeout cant be negative: {stabilisation_timeout}"
    assert depth_scale > 0, f"Depth scale should be positive: {depth_scale}"
    assert robot_positions.ndim == 2 and robot_positions.shape[0] > 0 and robot_positions.shape[1] == 7, f"Invalid robot joint positions: {robot_positions.shape}"

    depth_images = []
    bgr_images = []
    base_t_gripper_s = []

    align = rs.align(rs.stream.color)

    for frame_idx, position in enumerate(robot_positions):
        logging.info(f"moving to position {frame_idx} : {position}")

        reset_joints_to(robot_interface, position)
        time.sleep(stabilisation_timeout)

        base_t_gripper = robot_interface.last_eef_pose

        frames = image_pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        bgr_frame = np.ascontiguousarray(aligned_frames.get_color_frame().get_data())

        raw_depth_frame = aligned_frames.get_depth_frame()
        processed_depth_frame = depth_filter_pipeline.optimize_depth_image(raw_depth_frame)
        depth_frame_scaled = np.asanyarray(processed_depth_frame.get_data()) * depth_scale

        depth_images.append(depth_frame_scaled.copy())
        bgr_images.append(bgr_frame.copy())

        base_t_gripper_s.append(base_t_gripper)

    return depth_images, bgr_images, base_t_gripper_s


def gather_robot_data(
        number_of_positions:None|int = None,
        depth_filter_config:DepthOptimisationConfig = DepthOptimisationConfig(),
        stabilisation_timeout:float = 0.0,
        position_file:str = "positions_panda_personpov_19.csv"
    ) -> RawRobotScan:
    """
    Creates an instance of ProtoRobotData by moving the robot and taking images:

    :param number_of_positions: The number of positions to gather images for. If None it gathers images for all passed positions
    :param stabilisation_timeout: the time to wait after moving the robot before taking an image in seconds

    :return: an instance of ProtoRobotData
    """
    assert number_of_positions is None or number_of_positions > 0, f"Number of positions must be positive/no limit, is: {number_of_positions}"
    assert stabilisation_timeout >= 0, f"Timeout cant be negative: {stabilisation_timeout}"


    robot_positions = np.loadtxt(os.path.join(os.path.dirname(__file__), position_file), delimiter=",")
    assert robot_positions.ndim == 2 and robot_positions.shape[0] > 0 and robot_positions.shape[1] == 7, f"Invalid robot joint positions: {robot_positions.shape}"
    if number_of_positions is not None:
        robot_positions = robot_positions[:number_of_positions]

    robot_interface = FrankaInterface(config_root + "/charmander.yml", use_visualizer=False)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)

    pipeline.start(config)

    rgb_intrinsics = pipeline.get_active_profile().get_stream(
        rs.stream.color).as_video_stream_profile().get_intrinsics()

    rgb_cam_mat, rgb_cam_dist_coef = extract_intrinsics(rgb_intrinsics)
    depth_scale = pipeline.get_active_profile().get_device().first_depth_sensor().get_depth_scale()

    depth_filter_pipeline = DepthFilterPipeline(depth_filter_config)

    depth_images, bgr_images, base_t_gripper_s = gather_robot_imgs_eefs(robot_interface=robot_interface,
                                                                        image_pipeline=pipeline,
                                                                        depth_scale=depth_scale,
                                                                        robot_positions=robot_positions,
                                                                        stabilisation_timeout=stabilisation_timeout,
                                                                        depth_filter_pipeline = depth_filter_pipeline
                                                                        )
    pipeline.stop()
    robot_interface.close()
    gathered_data = RawRobotScan(
        depth_images=np.array(depth_images),
        cam_intrinsic_mtx=rgb_cam_mat,
        cam_distortion_coefficients=rgb_cam_dist_coef,
        bgr_images=np.array(bgr_images),
        base_t_gripper_s=np.array(base_t_gripper_s),
    )
    return gathered_data