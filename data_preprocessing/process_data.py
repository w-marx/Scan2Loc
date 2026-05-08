import numpy as np
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gathering.aruco_charuco_detection import *

from image_to_pointcloud import *
from load_and_save import *
import argparse


def process_data(
        input_folder: str = "./in_data",
        output_folder: str = "./out_data",
        number_of_sampled_datapoints: int = 10,
        only_sample_robot_datapoints_w_marker_estimates: bool = False,
        markers_use_advanced_removal: bool = False, #TODO
        est3d_use_map_anything: bool = True,
        est3d_use_sam3_for_foreground_seg: bool = True,
        est3d_xyz_img_custom_downscaling:bool = False, # TODO
        est3d_xyz_img_upscaling:bool = False, #TODO
        est3d_xyz_img_confidence_threshold: int = 10,
        est3d_pointcloud_foreground_masks_conf_threshold: float = 0.5,
        est3d_pointcloud_foreground_object_detection_threshold: float = 0.5,
        est3d_pointcloud_iforest_confidence_threshold: float = 0.0,
        est3d_use_Depth_images: bool = True,
        est3d_use_intrinsic_cam_mtx: bool = True,
        est3d_debug_pointcloud_visualize_result: bool = False,
        est3d_debug_visualize_foreground_masks: bool = False,
    ):
    """

    :param input_folder: The Folder path from which to load the data
    :param output_folder: The Folder path to which to save the data
    :return:
    """

    # Check that the parameters are valid
    assert number_of_sampled_datapoints > 0, "Non positive number of datapoints cant be sampled"
    assert 0 <= est3d_xyz_img_confidence_threshold <= 100, "est3d_xyz_img_confidence_threshold out of range: 0-100"
    assert 0.0 <= est3d_pointcloud_foreground_object_detection_threshold <= 1.0, "est3d_pointcloud_foreground_object_detection_threshold out of range: 0.0-1.0"
    assert 0.0 <= est3d_pointcloud_foreground_masks_conf_threshold <= 1.0, "est3d_pointcloud_foreground_masks_conf_threshold out of range: 0.0-1.0"
    assert 0.0 <= est3d_pointcloud_iforest_confidence_threshold <= 1.0, "est3d_pointcloud_iforest_confidence_threshold out of range: 0.0-1.0"


    print(f"Loading the data from {input_folder}...")
    loaded_data = load_input_data(input_folder=input_folder)

    headset_images:np.ndarray = loaded_data["headset_images"]
    headset_cam_mtx:np.ndarray = loaded_data["headset_cam_mtx"]
    headset_cam_dist_coeffs:list[float] = loaded_data["headset_dist_coef"]

    robot_images_names:list[str] = loaded_data["robot_folder_names"]
    robot_rgb_images:list[np.ndarray] = loaded_data["robot_rgb_images"]
    robot_depth_images:list[np.ndarray] = loaded_data["robot_depth_images"]
    robot_camera_t_marker_s:list[np.ndarray | None] = loaded_data["robot_camera_t_marker_s"]
    robot_base_t_robot_camera_s:list[np.ndarray] = loaded_data["robot_base_t_camera_s"]

    robot_rgb_cam_mtx:np.ndarray = loaded_data["robot_rgb_cam_mtx"]
    robot_rgb_cam_dist_coef:list[float] = loaded_data["robot_rgb_dist"]
    robot_depth_cam_mtx:np.ndarray = loaded_data["robot_depth_cam_mtx"]


    # Marker handling
    headset_image = headset_images[int(len(headset_images)/2)]
    headset_t_marker = None

    marker_detector:ArucoCharucoDetector|None = loaded_data["marker_detector"]
    if marker_detector is not None:
        # Select better headset image
        headset_t_markers = marker_detector.get_camera_t_marker(
            images=list(headset_images),
            camera_matrix=headset_cam_mtx,
            distortion_coefficients = robot_rgb_cam_dist_coef
        )
        headset_t_markers_idx_none_filtered = [idx for idx, h_t_m in enumerate(headset_t_markers) if h_t_m is not None]
        if len(headset_t_markers_idx_none_filtered) > 0:
            headset_w_marker_img_idx = min(headset_t_markers_idx_none_filtered, key = lambda x: np.abs(x-len(headset_images)/2))
            headset_t_marker = headset_t_markers[headset_w_marker_img_idx]
            headset_image = headset_images[headset_w_marker_img_idx]

        headset_image = marker_detector.remove_markers([headset_image])[0]
        robot_rgb_images = marker_detector.remove_markers(robot_rgb_images)


    # Choose the robot images smartly
    robot_image_indices_w_base_t_marker = [idx for idx, _ in enumerate(robot_camera_t_marker_s) if robot_base_t_robot_camera_s is not None]

    chosen_indices = [idx for idx, _ in enumerate(robot_rgb_images)]

    if number_of_sampled_datapoints <= len(robot_image_indices_w_base_t_marker) or only_sample_robot_datapoints_w_marker_estimates:
        chosen_indices = robot_image_indices_w_base_t_marker[:number_of_sampled_datapoints]
    else:
        indices_no_marker_pose = set(chosen_indices) - set(robot_image_indices_w_base_t_marker)
        chosen_indices = robot_image_indices_w_base_t_marker + indices_no_marker_pose[:number_of_sampled_datapoints-len(robot_image_indices_w_base_t_marker)]

    print(f"length of chosen indices {len(chosen_indices)}")

    robot_rgb_images = [robot_rgb_images[i] for i in chosen_indices]
    robot_depth_images = [robot_depth_images[i] for i in chosen_indices]
    robot_camera_t_marker_s = [robot_camera_t_marker_s[i] for i in chosen_indices]
    robot_base_t_robot_cameras_s = [robot_base_t_robot_camera_s[i] for i in chosen_indices]
    robot_images_names = [robot_images_names[i] for i in chosen_indices]

    # Generate 3D Point cloud
    print("Generating point cloud...")
    robot_base_xyz_imgs = None
    point_cloud = None
    image_mask_generator = lambda imgs: create_foreground_masks(
                images=imgs,
                threshold=est3d_pointcloud_foreground_object_detection_threshold,
                mask_threshold = est3d_pointcloud_foreground_masks_conf_threshold,
                visualize_masks=est3d_debug_visualize_foreground_masks
    ) if est3d_use_sam3_for_foreground_seg else None

    if est3d_use_map_anything:
        robot_rgb_images, robot_base_xyz_imgs, point_cloud, robot_rgb_cam_mtx = create_point_cloud(
            rgb_images=np.array(robot_rgb_images),
            base_t_cam_s=np.array(robot_base_t_robot_cameras_s),
            depth_images=np.array(robot_depth_images) if est3d_use_Depth_images else None,
            camera_intrinsics=robot_rgb_cam_mtx if est3d_use_intrinsic_cam_mtx else None,
            confidence_threshold_percent=est3d_xyz_img_confidence_threshold,
            image_mask_generator=image_mask_generator,
            visualize_point_cloud=est3d_debug_pointcloud_visualize_result
        )
        # Use Iforest on pointcloud
        point_cloud = remove_outliers_from_point_cloud(
            points=point_cloud,
            contamination=est3d_pointcloud_iforest_confidence_threshold
        )
    else:
        robot_base_xyz_imgs, point_cloud = create_point_cloud_simple(
            depth_images=np.array(robot_depth_images),
            depth_cam_mtx=np.array(robot_depth_cam_mtx),
            base_t_camera_s=np.array(robot_base_t_robot_cameras_s),
            image_masks=image_mask_generator(np.array(robot_rgb_images)) if image_mask_generator else None,
            distance_cutoff=1.0,
            visualize_point_cloud=True
        )

    print("Generating the labels...")
    robot_base_t_headsets = [None] * len(robot_rgb_images)
    if headset_t_marker is not None:
        for idx, (robot_base_t_camera, robot_camera_t_marker) in enumerate(zip(robot_base_t_robot_camera_s, robot_camera_t_marker_s)):
            if robot_camera_t_marker is not None:
                robot_base_t_headsets[idx] = robot_base_t_camera @ robot_camera_t_marker @ np.linalg.inv(headset_t_marker)


    print("Saving the data...")
    save_output_data(
        output_folder=output_folder,
        headset_image=headset_image,
        headset_cam_mtx=headset_cam_mtx,
        robot_folder_names=robot_images_names,
        robot_rgb_cam_mtx=robot_rgb_cam_mtx,
        robot_rgb_images=np.array(robot_rgb_images),
        robot_xyz_images=robot_base_xyz_imgs,
        point_cloud = point_cloud,
        robot_base_t_robot_camera_s = robot_base_t_robot_camera_s,
        robot_base_t_headsets = robot_base_t_headsets,
        headset_cam_dist_coeffs = headset_cam_dist_coeffs,
        robot_rgb_cam_dist_coeffs = robot_rgb_cam_dist_coef
    )



if __name__ == "__main__":
    #TODO implement good parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, default="../in_folder", help="Input Folder Location")
    parser.add_argument("--output-folder", type=str, default="./out_data", help="Output Folder Location")

    args = parser.parse_args()

    start_time = time.perf_counter()
    process_data(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        est3d_debug_pointcloud_visualize_result = False,
        number_of_sampled_datapoints = 10,
        est3d_pointcloud_iforest_confidence_threshold = 0.1,
        est3d_use_Depth_images= False,
        est3d_use_map_anything = False,
    )
    print(f"Data processing took {(time.perf_counter() - start_time):.6f} seconds")





