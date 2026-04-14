from image_to_pointcloud import *
from aruco_marker_handling import *
from load_and_save import *
import argparse


def mask_images(images:np.ndarray)->np.ndarray:
    """
    Takes the images and replaces found aruco markers with black pixels
    :param images: NxWxHx3 RGB images as an numpy array
    :return NxWxHx3 RGB images as an numpy array with the markers removed
    """
    masked_images = []
    for image in images:
        aruco_mask = get_marker_mask(image).astype(np.uint8)
        aruco_mask = np.stack([aruco_mask, aruco_mask, aruco_mask], axis=2)
        masked_images.append(image * aruco_mask)
    return np.array(masked_images)


def process_data(
        input_folder: str = "./in_data_vrs",
        output_folder: str = "./out_data",
        confidence_threshhold: float = 0.1,
        visualize_pointcloud: bool = True,
        calibration_board_size:tuple[int, int] = (9,6),
        calibration_board_square_size:float = 0.03,
        aruco_marker_size:float = 0.1,
        point_cloud_creation_method:str = "3D points",
    ):
    """

    :param input_folder: The Folder path from which to load the data
    :param output_folder: The Folder path to which to save the data
    :param confidence_threshhold: The bottom quantile of points that will be discarded when generating the 3D point cloud
    :param visualize_pointcloud: Wheater to open an tab to visualize the point cloud
    :param calibration_board_size: The dimensions of the calibration board in number of squares
    :param calibration_board_square_size: The size of a square on the calibration board in meters
    :param aruco_marker_size: The Side length of an marker in meters
    :param point_cloud_creation_method: `3D points` or `Depth`
    :param undistort_images: wheather to undistort the images or to continue with the distorted
    :return:
    """

    print("Loading the data...")
    headset_image, headset_calibration_images, robot_images, robot_image_names, robot_calibration_images = load_input_data(input_folder)

    print("Computing the camera properties...")
    headset_cam_mtx, headset_cam_dist_coef = calculate_camera_params_from_images(headset_calibration_images,calibration_board_size,calibration_board_square_size)
    robot_cam_mtx, robot_cam_dist_coef = calculate_camera_params_from_images(robot_calibration_images,calibration_board_size,calibration_board_square_size)


    ### Mask images:
    print("Masking images...")
    masked_robot_images = mask_images(robot_images)
    masked_headset_image = mask_images(np.array([headset_image]))[0]

    print("Compute headset poses...")
    headset_t_aruco = estimate_camera_aruco_pose(np.array([headset_image]), headset_cam_mtx, headset_cam_dist_coef, marker_side_length=aruco_marker_size)[0]


    #### create and Fill Robot folder
    print("Generating point cloud...")
    # Use vggt to create image points
    robot_imgs_3d_points, robot_imgs_3d_points_conf, robot_extrinsic, robot_intrinsic, robot_imgs_depth, robot_imgs_depth_conf= use_vggt_on_images(robot_images)

    # TODO using the real parameters doesnt improve performance
    #real_intrinsic = np.array([robot_cam_mtx for _ in range(np.shape(robot_intrinsic)[0])])
    #print(f"Real intrinsics shape: {real_intrinsic.shape}, fake : {robot_intrinsic.shape}")

    point_cloud = create_point_cloud_from_image_points(
        method=point_cloud_creation_method,
        images_points_3d_and_conf=(robot_imgs_3d_points, robot_imgs_3d_points_conf),
        images_depth_maps_and_conf=(robot_imgs_depth, robot_imgs_depth_conf, robot_extrinsic, robot_intrinsic),
        masks= create_foreground_masks(images=robot_images),
        visualize=visualize_pointcloud,
        confidence_quantile=confidence_threshhold,
    )

    print("Generating the labels...")
    # Compute ground truth pose estimates
    robot_cams_t_aruco = estimate_camera_aruco_pose(robot_images, robot_cam_mtx, robot_cam_dist_coef, aruco_marker_size)
    # Create RGB-XYZ image pairs and ground truth poses for the robot
    robot_poses_t_headset = [robot_cam_t_aruco @ np.linalg.inv(headset_t_aruco) for robot_cam_t_aruco in robot_cams_t_aruco]

    print("Saving the data...")
    save_output_data(
        output_folder=output_folder,
        headset_image=masked_headset_image,
        headset_cam_mtx=headset_cam_mtx,
        headset_cam_dist_coef=headset_cam_dist_coef,
        robot_image_names=robot_image_names,
        robot_cam_mtx=robot_cam_mtx,
        robot_cam_dist_coef=robot_cam_dist_coef,
        robot_rgb_images=masked_robot_images,
        robot_xyz_images=robot_imgs_3d_points,
        robot_cams_t_headset=robot_poses_t_headset,
        point_cloud = point_cloud
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, default="./in_data_vrs", help="Input Folder Location")
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
        calibration_board_size=args.calibration_board_size,
        calibration_board_square_size=args.calibration_board_square_size,
        aruco_marker_size=args.aruco_marker_size,
    )





