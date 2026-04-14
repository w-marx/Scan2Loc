import numpy as np

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
        robot_image_limit:int = 4
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
    :param robot_image_limit: The number of images selected from the robot images
    :return:
    """

    print(f"Loading the data from {input_folder}...")
    loaded_data = load_input_data(input_folder=input_folder)
    headset_images:np.ndarray = loaded_data["headset_images"]

    headset_cam_mtx:np.ndarray | None = loaded_data["headset_cam_mtx"]
    if headset_cam_mtx is None:
        headset_calibration_images = loaded_data["headset_calibration_images"]
        headset_cam_mtx, _ = calculate_camera_params_from_images(headset_calibration_images, calibration_board_size, calibration_board_square_size)

    robot_rgb_images:np.ndarray = loaded_data["robot_rgb_images"]
    robot_depth_images:np.ndarray = loaded_data["robot_depth_images"]
    robot_images_names:list[str] = loaded_data["robot_images_names"]

    robot_rgb_cam_mtx:np.ndarray = loaded_data["robot_rgb_cam_mtx"]
    robot_rgb_cam_dist_coef:np.ndarray = loaded_data["robot_rgb_dist"]

    robot_base_t_robot_cameras = loaded_data["robot_base_t_cameras"]


    # TODO choose the image smarter
    print(f"headset_images: {headset_images.shape}")
    headset_image:np.ndarray = headset_images[int(headset_images.shape[0]/2)]

    # TODO Filter the Robot images smarter
    # Is done to reduce memory / runtime limitations
    print(f"reducing the number of robot images...")
    indices = np.arange(start=1, stop=robot_image_limit+1)*int(len(robot_images_names)/(robot_image_limit+1))
    robot_rgb_images = robot_rgb_images[indices]
    robot_depth_images = robot_depth_images[indices]
    robot_images_names = [robot_images_names[i] for i in indices]

    ### Mask images:
    print("Masking images...")
    masked_robot_images = mask_images(robot_rgb_images)
    masked_headset_image = mask_images(np.array([headset_image]))[0]

    print("Compute headset poses...")
    headset_t_aruco = estimate_camera_aruco_pose(np.array([headset_image]), headset_cam_mtx, np.zeros(shape = 5), marker_side_length=aruco_marker_size)[0]


    #### create and Fill Robot folder
    print("Generating point cloud...")
    # Use vggt to create image points
    robot_imgs_3d_points, robot_imgs_3d_points_conf, robot_extrinsic, robot_intrinsic, robot_imgs_depth, robot_imgs_depth_conf= use_vggt_on_images(robot_rgb_images)

    point_cloud = create_point_cloud_from_image_points(
        method=point_cloud_creation_method,
        images_points_3d_and_conf=(robot_imgs_3d_points, robot_imgs_3d_points_conf),
        images_depth_maps_and_conf=(robot_imgs_depth, robot_imgs_depth_conf, robot_extrinsic, robot_intrinsic),
        masks= create_foreground_masks(images=robot_rgb_images),
        visualize=visualize_pointcloud,
        confidence_quantile=confidence_threshhold,
    )

    print("Adjusting the vggt scale...")
    if robot_base_t_robot_cameras is not None:
        scale = 1
        transformation_unscaled = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

        print("Adjusting the vggt scale based on the given camera poses")
        # TODO Add support if when they are actually provided by the Robot
    else:
        print("Defaulting to vggt extrinsics which are not scaled")
        print("ONLY USABLE FOR PROOF OF CONCEPT TESTING")
        robot_base_t_robot_cameras = [np.append(robot_extrinsic[i], np.array([[0,0,0,1]]), axis=0) for i in range(robot_extrinsic.shape[0])]
        print(robot_base_t_robot_cameras)


    print("Generating the labels...")

    robot_cams_t_aruco = estimate_camera_aruco_pose(robot_rgb_images, robot_rgb_cam_mtx, robot_rgb_cam_dist_coef, aruco_marker_size)
    # Create RGB-XYZ image pairs and ground truth poses for the robot

    robot_cams_t_headset = [robot_cam_t_aruco @ np.linalg.inv(headset_t_aruco) for robot_cam_t_aruco in robot_cams_t_aruco]

    robot_base_t_headsets = [robot_base_t_robot_cameras[i] @ robot_cams_t_headset[i] for i in range(len(robot_cams_t_headset))]


    print("Saving the data...")
    save_output_data(
        output_folder=output_folder,
        headset_image=masked_headset_image,
        headset_cam_mtx=headset_cam_mtx,
        robot_image_names=robot_images_names,
        robot_rgb_cam_mtx=robot_rgb_cam_mtx,
        robot_rgb_images=masked_robot_images,
        robot_xyz_images=robot_imgs_3d_points,
        point_cloud = point_cloud,
        robot_base_t_robot_cameras = robot_base_t_robot_cameras,
        robot_base_t_headsets = robot_base_t_headsets,
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, default="./input_data_examples/Example1_w_aruco", help="Input Folder Location")
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





