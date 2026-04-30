import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_gathering.aruco_charuco_detection import *

from image_to_pointcloud import *
from load_and_save import *
import argparse


def process_data(
        input_folder: str = "./in_data_vrs",
        output_folder: str = "./out_data",
        confidence_threshhold: float = 0.1,
        iforest_confidence_threshhold: float = 0.05,
        visualize_pointcloud: bool = True,
        point_cloud_creation_method:str = "3D points",
        robot_image_limit:int = 1
    ):
    """

    :param input_folder: The Folder path from which to load the data
    :param output_folder: The Folder path to which to save the data
    :param confidence_threshhold: The bottom quantile of points that will be discarded when generating the 3D point cloud
    :param iforest_confidence_threshhold: The percentage of points removed by iforest during point cloud generation
    :param visualize_pointcloud: Wheater to open an tab to visualize the point cloud
    :param calibration_board_size: The dimensions of the calibration board in number of squares
    :param calibration_board_square_size: The size of a square on the calibration board in meters
    :param aruco_marker_size: The Side length of an marker in meters
    :param point_cloud_creation_method: `3D points` or `Depth`
    :param robot_image_limit: The number of images selected from the robot images
    :return:
    """

    print(f"Loading the data from {input_folder}...")
    loaded_data = load_input_data(input_folder=input_folder)

    headset_images:np.ndarray = loaded_data["headset_images"]
    headset_cam_mtx:np.ndarray = loaded_data["headset_cam_mtx"]

    robot_rgb_images:list[np.ndarray] = loaded_data["robot_rgb_images"]
    robot_depth_images:list[np.ndarray] = loaded_data["robot_depth_images"]
    robot_camera_t_marker:list[np.ndarray | None] = loaded_data["robot_camera_t_marker_s"]
    robot_base_t_robot_cameras:list[np.ndarray] = loaded_data["robot_base_t_camera_s"]
    robot_images_names:list[str] = loaded_data["robot_images_names"]

    robot_rgb_cam_mtx:np.ndarray = loaded_data["robot_rgb_cam_mtx"]
    robot_rgb_cam_dist_coef:np.ndarray = loaded_data["robot_rgb_dist"]


    # TODO choose the image smarter
    print(f"headset_images: {headset_images.shape}")
    headset_image:np.ndarray = headset_images[int(headset_images.shape[0]/2)]

    print("Masking images...")
    aruco_charuco_detector:ArucoCharucoDetector|None = loaded_data["marker_detector"]
    masked_headset_image = headset_image
    masked_robot_images = robot_rgb_images
    if aruco_charuco_detector is not None:
        masked_headset_image = aruco_charuco_detector.remove_markers([headset_image])[0]
        masked_robot_images = aruco_charuco_detector.remove_markers(robot_rgb_images)


    #### create and Fill Robot folder
    print("Generating point cloud...")
    # Use vggt to create image points
    robot_imgs_3d_points, robot_imgs_3d_points_conf, robot_extrinsic, robot_intrinsic, robot_imgs_depth, robot_imgs_depth_conf= use_vggt_on_images(np.array(robot_rgb_images[:robot_image_limit]))

    point_cloud = create_point_cloud_from_image_points(
        method=point_cloud_creation_method,
        images_points_3d_and_conf=(robot_imgs_3d_points, robot_imgs_3d_points_conf),
        images_depth_maps_and_conf=(robot_imgs_depth, robot_imgs_depth_conf, robot_extrinsic, robot_intrinsic),
        masks= create_foreground_masks(images=np.array(robot_rgb_images[:robot_image_limit])),
        visualize=visualize_pointcloud,
        confidence_quantile=confidence_threshhold,
        iforest_quantile=iforest_confidence_threshhold
    )

    print("Adjusting the vggt scale...")
    #TODO Add support for rescaling the vggt extrinsics

    print("Generating the labels...")
    robot_base_t_headsets = [None] * len(robot_rgb_images)
    if aruco_charuco_detector is not None:
        headset_t_marker = aruco_charuco_detector.get_camera_t_marker([headset_image], headset_cam_mtx, [0,0,0,0,0])[0]
        if headset_t_marker is not None:
            for idx, (robot_base_t_camera, robot_camera_t_marker) in enumerate(zip(robot_base_t_robot_cameras, robot_camera_t_marker)):
                if robot_camera_t_marker is not None:
                    robot_base_t_headsets[idx] = robot_base_t_camera @ robot_camera_t_marker @ np.linalg.inv(headset_t_marker)



    print("Saving the data...")
    save_output_data(
        output_folder=output_folder,
        headset_image=masked_headset_image,
        headset_cam_mtx=headset_cam_mtx,
        robot_image_names=robot_images_names,
        robot_rgb_cam_mtx=robot_rgb_cam_mtx,
        robot_rgb_images=np.array(masked_robot_images),
        robot_xyz_images=robot_imgs_3d_points,
        point_cloud = point_cloud,
        robot_base_t_robot_cameras = robot_base_t_robot_cameras,
        robot_base_t_headsets = robot_base_t_headsets,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, default="../gathered_data_examples/aruco1", help="Input Folder Location")
    parser.add_argument("--output-folder", type=str, default="./out_data", help="Output Folder Location")
    parser.add_argument("--confidence-threshhold", type=float, default=0.1, help="Confidence threshold for points in the 3D point cloud")
    parser.add_argument("--visualize-pointcloud", type=bool, default=True, help="If the point cloud is to be visualized in a window")
    parser.add_argument("--calibration-board-size", type=tuple[int, int], default=(9,6), help="Calibration Board size in meters")
    parser.add_argument("--calibration-board-square-size", type=float, default=0.03, help="Calibration Board square size in meters")
    parser.add_argument("--aruco-marker-size", type=float, default=0.1, help="Aruco marker size in meters")

    args = parser.parse_args()

    process_data(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        confidence_threshhold=args.confidence_threshhold,
        visualize_pointcloud=args.visualize_pointcloud,
    )





