import os
import numpy as np
import cv2
import json
import argparse
import colorsys
import shutil

import open3d as o3d
import open3d.visualization.gui as gui

import matplotlib.pyplot as plt

from data_gathering.aruco_charuco_detection import ArucoCharucoDetector, ArucoDetector, CharucoDetector

DICTIONARY_OPTIONS = {
    "5X5_100":cv2.aruco.DICT_5X5_100,
    "5X5_250":cv2.aruco.DICT_5X5_250,
    "6X6_250":cv2.aruco.DICT_6X6_250,
    "7X7_250":cv2.aruco.DICT_7X7_250,
    "7X7_1000":cv2.aruco.DICT_7X7_1000,
}

def optimize_robot_data(
        rgb_images: list[np.ndarray],
        base_t_gripper_s: list[np.ndarray],
        rgb_cam_mat: np.ndarray,
        rgb_cam_dist_coef: list[float],
        output_folder:str = "data",
        marker_detector: ArucoCharucoDetector | None = None,
        gripper_t_cam: np.ndarray|None = None,
    ):
    """
    Creates the following output folder format by moving the robot and taking images:

    `output_folder`
    ├── robot
    │   └── multiple folders with the contents:
    │       ├──  A rgb.png image
    │       ├──  A poses.json file
    │       └──  A depth.npy file
    └── robot_cam_calibration.json

    :param rgb_images: list of WxHx3-uint8 rgb images
    :param gripper_t_cam: A 4x4 transformation matrix for gripper^T_Cam, if None will be estimated
    :param marker_detector: A aruco/charuco marker detector depending on what will be seen in the images
                            or none if they should not be estimated (but then gripper_t_cam must be provided)
    :param rgb_cam_dist_coef: distortion coefficients for the camera
    :param rgb_cam_mat: intrinsic camera matrix
    :param base_t_gripper_s: list of 4x4 transformation matrices for the robot base^T_gripper
    :param output_folder: the name of the output folder
    :return: nothing
    """

    if gripper_t_cam is None and marker_detector is None:
        raise ValueError("Either gripper_t_cam or marker_detector must be provided")

    camera_t_marker_s = [None] * len(rgb_images)
    if marker_detector is not None:
        camera_t_marker_s = marker_detector.get_camera_t_marker(images=rgb_images, camera_matrix=rgb_cam_mat, distortion_coefficients=rgb_cam_dist_coef)

    if gripper_t_cam is None:
        print("Using aruco markers for gripper_T_cam determination")

        r_base_t_gripper , t_base_t_gripper, r_marker_t_camera, t_marker_t_camera = [], [], [], []

        for camera_t_marker, base_t_gripper in zip(camera_t_marker_s, base_t_gripper_s):
            if camera_t_marker is not None:
                r_base_t_gripper.append(base_t_gripper[:3, :3])
                t_base_t_gripper.append(base_t_gripper[:3, 3])
                r_marker_t_camera.append(camera_t_marker[:3, :3])
                t_marker_t_camera.append(camera_t_marker[:3, 3])

        if len(r_base_t_gripper) < 5:
            print(f"Dangerously few aruco marker images for hand eye calibration: {len(r_base_t_gripper)}")

        r_gripper_t_cam, t_gripper_t_cam = cv2.calibrateHandEye(r_base_t_gripper, t_base_t_gripper, r_marker_t_camera, t_marker_t_camera, method=cv2.CALIB_HAND_EYE_DANIILIDIS)
        gripper_t_cam = np.concatenate((np.concatenate((r_gripper_t_cam, t_gripper_t_cam), axis=1), [[0, 0, 0, 1]]), axis=0)

    for i, (rgb_image, base_t_gripper) in enumerate(zip(rgb_images, base_t_gripper_s)):
        os.makedirs(f"{output_folder}/robot/{i:06d}", exist_ok=True)

        pose_dict = {
            "base_t_gripper": base_t_gripper.tolist(),
            "gripper_t_cam": gripper_t_cam.tolist(),
            "base_t_cam": (base_t_gripper @ gripper_t_cam).tolist(),
            "camera_t_marker": camera_t_marker_s[i].tolist() if camera_t_marker_s[i] is not None else None,
        }
        with open(f"{output_folder}/robot/{i:06d}/poses.json", 'w') as f:
            json.dump(pose_dict, f, indent=4)


def visualize_poses(
        base_t_gripper_s:list[np.ndarray],
        gripper_t_camera_s:list[np.ndarray],
        camera_t_aruco_s:list[np.ndarray|None],
        table_dimensions:tuple[float, float, float] = (1.5, 1.5, 0.05),
        aruco_marker_dimensions:tuple[float, float, float]  = (0.072, 0.072, 0.001)
    ):
    """
    Visualizes the poses in O3D
    """

    app = gui.Application.instance
    app.initialize()

    vis = o3d.visualization.O3DVisualizer("Pose Visualizer", 1280, 720)

    # Create table & Robot Base
    origin_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    vis.add_geometry("origin", origin_frame)
    vis.add_3d_label(np.array([0,0,0]), f"Robot Base")

    table = o3d.geometry.TriangleMesh.create_box(width = table_dimensions[0],height=table_dimensions[1],depth=table_dimensions[2])
    table.translate([-table_dimensions[0]/2, -table_dimensions[1]/2, -table_dimensions[2]])
    vis.add_geometry("table", table)


    for i, (base_t_gripper, gripper_t_camera, camera_t_aruco) in enumerate(zip(base_t_gripper_s, gripper_t_camera_s, camera_t_aruco_s)):

        color = colorsys.hsv_to_rgb(i/(len(base_t_gripper_s)+1), 1.0, 1.0)

        # Add the gripper frame + sphere

        gripper_cord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
        gripper_cord_frame.transform(base_t_gripper)
        vis.add_geometry(f"gripper_frame_{i}", gripper_cord_frame)

        gripper_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
        gripper_sphere.translate(base_t_gripper[:3, 3])
        gripper_sphere.paint_uniform_color(color)
        vis.add_geometry(f"gripper_sphere_{i}", gripper_sphere)
        vis.add_3d_label(base_t_gripper[:3, 3], f"G{i}")


        # Add the camera frame + sphere

        base_t_camera = base_t_gripper @ gripper_t_camera


        camera_cord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.04)
        camera_cord_frame.transform(base_t_camera)
        vis.add_geometry(f"camera_frame{i}", camera_cord_frame)

        camera_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        camera_sphere.translate(base_t_camera[:3, 3])
        camera_sphere.paint_uniform_color(color)
        vis.add_geometry(f"camera_sphere_{i}", camera_sphere)
        vis.add_3d_label(base_t_camera[:3, 3], f"C{i}")



        # --- Aruco Frame ---
        if camera_t_aruco is not None:
            base_t_aruco = base_t_camera @ camera_t_aruco

            aruco_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.02)
            aruco_frame.transform(base_t_aruco)
            vis.add_geometry(f"aruco_frame{i}", aruco_frame)

            marker = o3d.geometry.TriangleMesh.create_box(
                width=aruco_marker_dimensions[0],
                height=aruco_marker_dimensions[1],
                depth=aruco_marker_dimensions[2]
            )
            marker.translate([-aruco_marker_dimensions[0] / 2, -aruco_marker_dimensions[1] / 2, -aruco_marker_dimensions[2]])
            marker.transform(base_t_aruco)
            marker.paint_uniform_color(color)
            vis.add_geometry(f"aruco_marker{i}", marker)
            vis.add_3d_label(base_t_aruco[:3, 3], f"A{i}")

    app.add_window(vis)
    app.run()



def check_output_data(output_folder:str = "data", visualize:bool = True, aruco_size:float = 0.072):
    print("checking results")
    json_files = []
    for folder in os.listdir(f"{output_folder}/robot"):
        with open(f"{output_folder}/robot/{folder}/poses.json", 'r') as f:
            json_files.append(json.load(f))


    base_t_gripper_s = np.array([pose["base_t_gripper"] for pose in json_files])
    gripper_t_camera_s = np.array([pose["gripper_t_cam"] for pose in json_files])
    camera_t_marker_s = [(np.array(pose["camera_t_aruco"]) if pose["camera_t_aruco"] is not None else None) for pose in json_files]

    if visualize:
        visualize_poses(
            base_t_gripper_s = base_t_gripper_s,
            gripper_t_camera_s = gripper_t_camera_s,
            camera_t_aruco_s = camera_t_marker_s,
            table_dimensions = (1.5, 1.5, 0.05),
            aruco_marker_dimensions = (aruco_size, aruco_size, 0.001)
        )


    # Evaluating base_T_aruco performance
    print(f"{len([p for p in camera_t_marker_s if p is not None])}/{len(camera_t_marker_s)} positions have aruco pose estimates")

    base_t_aruco_s = [r_t_g @ g_t_c @ c_t_a for r_t_g, g_t_c, c_t_a in zip(base_t_gripper_s, gripper_t_camera_s, camera_t_marker_s) if c_t_a is not None]

    avg_aruco_xyz_position = np.mean(np.array([base_t_aruco[:3, 3] for base_t_aruco in base_t_aruco_s]), axis = 0)


    calc_translat_difference = lambda x,y: np.linalg.norm(x - y)
    calc_rotational_difference = lambda x, y: np.arccos((np.trace(x[:3, :3] @ y[:3, :3].T)-1)/2)



    avg_translat_error = np.mean([np.linalg.norm(base_t_aruco[:3, 3]-avg_aruco_xyz_position) for base_t_aruco in base_t_aruco_s])

    print(f"Average marker xyz position in frame R: {np.round(avg_aruco_xyz_position, 4)}m")
    print(f"Avg translational error: {np.round(avg_translat_error*1000, 2)}mm")

    # plot results:
    fig, axes = plt.subplots(2,2, figsize = (10, 10))
    
    translat_differences = [calc_translat_difference(b_t_a[:3,3],avg_aruco_xyz_position)*1000 for b_t_a in base_t_aruco_s]
    axes[0,0].boxplot(translat_differences)
    axes[0,0].set_title(f'Translational error (avg: {np.round(avg_translat_error*1000, 2)} mm)')
    axes[0,0].set_ylabel(f'Translational deviation from Average in mm')
    x = np.ones(len(translat_differences))
    axes[0,0].scatter(np.ones(len(translat_differences)), translat_differences, alpha=0.6)


    axes[0,1].set_title(f'Estimated translational positions:')
    x_positions = [b_t_a[0,3] for b_t_a in base_t_aruco_s]
    y_positions = [b_t_a[1,3] for b_t_a in base_t_aruco_s]
    z_positions = [b_t_a[2,3] for b_t_a in base_t_aruco_s]

    xyz_pos_plot = axes[0,1].scatter(
        x_positions,y_positions, 
        c = z_positions,
        cmap = 'viridis',
        s = 50,
        alpha = 0.6
    )


    axes[0,1].set_xlabel("Estimated x pos in meter")
    axes[0,1].set_ylabel("Estimated y pos in meter")


    cbar = plt.colorbar(xyz_pos_plot, ax=axes[0,1])
    cbar.set_label('Z position in meters', fontsize=10)

    plt.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", type=str, default="my_data", help="Output Folder Location")
    parser.add_argument("--cam-t-gripper-path", type=str, default=None, help="Path to cam_t_gripper.npy file, if left to None will be estimated")


    parser.add_argument("--marker_detection", type = "str", default=None, help = "If Aruco / Charuco marker detection should be used, options: `None`(default), `Aruco`, `Charuco`")
    parser.add_argument("--aruco-marker-side-length", type=float, default=0.072, help="Aruco marker side lengths in meters")
    parser.add_argument("--aruco-marker-dictionary", type=str, default="6X6_250", help=f"Aruco dictionary to use, possible options are: {', '.join(DICTIONARY_OPTIONS.keys())}")
    parser.add_argument("--charuco-square-side-length", type=float, default=0.1, help="Charuco board square side length in meters")
    parser.add_argument("--charuco-board-size", type=int, default=[14,9], nargs=2, help="Size of the charuco board in squares")

    parser.add_argument("--no-data-gathering", action = "store_false", help = "If used only optimization & evaluation may be done", dest = "gather_data")
    parser.add_argument("--stabilisation-timeout", type=float, default=0.0, help="Timeout in seconds between robot moved to position and picture is taken")

    parser.add_argument("--no-3D-visualize", action = "store_false", help = "If used there wont be any 3D pose visualisation", dest = "visualize_poses")
    parser.add_argument("--no-result-analysation", action = "store_false", help = "If used there wont by any result analysation (pose deviation analysis)", dest = "analyze_results")

    parser.set_defaults(gather_data = True, visualize_poses = True, analyze_results = True)
    args = parser.parse_args()

    print(f"Saving/loading data from: {os.path.abspath(args.output_folder)}")

    cam_t_gripper = None
    if args.cam_t_gripper_path is not None and os.path.exists(args.cam_t_gripper_path):
        cam_t_gripper = np.load(args.cam_t_gripper_path)
        print(f"Using precomputed cam_t_gripper: \n {np.round(cam_t_gripper, 3)} \n from {args.cam_t_gripper_path}")


    rgb_images, base_t_gripper_s, rgb_cam_mat, rgb_cam_dist_coef = None, None, None, None
    if args.gather_data:
        print("gathering data using the robot...")
        if os.path.exists(f"{args.output_folder}"):
            print(f"Output folder already exists, deleting it ...")
            shutil.rmtree(f"{args.output_folder}")

        from data_gathering.robot_interface import gather_robot_data
        rgb_images, base_t_gripper_s, rgb_cam_mat, rgb_cam_dist_coef = gather_robot_data(output_folder=args.output_folder)
    else:
        print("loading data from disk for further processing ...")
        folders = sorted(os.listdir(f"{args.output_folder}/robot"))

        rgb_images = [cv2.imread(f"{args.output_folder}/robot/{folder}/rgb.png") for folder in folders]

        base_t_gripper_s = []

        for folder in folders:
            with open(f"{args.output_folder}/robot/{folder}/poses.json", 'r') as f:
                base_t_gripper_s.append(np.array(json.load(f)["base_t_gripper"]))

        with open(f"{args.output_folder}/robot_cam_calibration.json", 'r') as f:
            json_file = json.load(f)
            rgb_cam_mat = np.array(json_file["rgb_camera_matrix"])
            rgb_cam_dist_coef = json_file["rgb_distortion_coefficients"]


    marker_detector = None

    if args.marker_detection is not None and args.marker_detection == "Aruco":
        marker_detector = ArucoDetector(
            aruco_marker_side_length=args.aruco_marker_side_length,
            aruco_marker_dictionary=DICTIONARY_OPTIONS[args.aruco_dictionary]
        )
    if args.marker_detection is not None and args.marker_detection == "Charuco":
        marker_detector = CharucoDetector(
            board_size=(args.charuco_board_size[0], args.charuco_board_size[1]),
            square_size=args.charuco_square_side_length,
            marker_size=args.aruco_marker_side_length,
            aruco_dictionary=DICTIONARY_OPTIONS[args.aruco_dictionary]
        )

    optimize_robot_data(
        rgb_images=rgb_images,
        base_t_gripper_s=base_t_gripper_s,
        rgb_cam_mat=rgb_cam_mat,
        rgb_cam_dist_coef=rgb_cam_dist_coef,
        output_folder=args.output_folder,
        gripper_t_cam=cam_t_gripper,
    )

    if args.analyze_results:
        print(f"analyzing data with 3D pose visualisation: {args.visualize_poses}")
        check_output_data(
            output_folder = args.output_folder,
            visualize = args.visualize_poses,
            aruco_size = args.aruco_marker_size 
        )
    print("main finished")
