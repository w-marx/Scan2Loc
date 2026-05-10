import os, cv2, json, argparse, shutil, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aruco_charuco_detection import ArucoCharucoDetector, ArucoDetector, CharucoDetector
from gathering_2_preprocessing import compute_pose_pseudo_median, GatheredRobotData

calc_rotational_difference = lambda x, y: np.arccos((np.trace(x[:3, :3] @ y[:3, :3].T) - 1) / 2)



def optimize_robot_data(
        bgr_images: list[np.ndarray],
        base_t_gripper_s: list[np.ndarray],
        color_cam_mat: np.ndarray,
        color_cam_dist_coef: list[float],
        depth_images: np.ndarray,
        depth_cam_mat: np.ndarray,
        depth_cam_distortion_coefficients: list[float],
        name: str = "rob_data1",
        marker_detector: ArucoCharucoDetector | None = None,
        gripper_t_cam: np.ndarray|None = None,
        base_t_gripper_outlier_quantiles:tuple[float, float] = (0.2, 0.2)
    )->GatheredRobotData:
    """
    Processed Raw gathered robot data to get a GatheredRobotData instance

    :param bgr_images: list of HxWx3-uint8 bgr images
    :param gripper_t_cam: A 4x4 transformation matrix for gripper^T_Cam, if None will be estimated
    :param marker_detector: A aruco/charuco marker detector depending on what will be seen in the images
                            or none if they should not be estimated (but then gripper_t_cam must be provided)
    :param color_cam_dist_coef: distortion coefficients for the color camera
    :param color_cam_mat: intrinsic camera matrix
    :param depth_images: A NxHxW-float array of depth images
    :param depth_cam_mat: The intrinsic 3x3 matrix of the depth camera
    :param depth_cam_distortion_coefficients: The distortion coefficients of the depth camera
    :param base_t_gripper_s: list of 4x4 transformation matrices for the robot base^T_gripper
    :param name: the name of the output folder/ dataset
    :param base_t_gripper_outlier_quantiles the quantiles of base_t_gripper estimates to remove, first float for translation and second for rotation
    :return: a GatheredRobotData instance
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

    # Remove camera_t_marker estimates that lead to outliers
    b_t_m_s = [b_t_g @ gripper_t_cam @ c_t_m for b_t_g, c_t_m in zip(base_t_gripper_s, camera_t_marker_s) if
               c_t_m is not None]
    
    if len(b_t_m_s) > 1:
        base_t_marker_median = compute_pose_pseudo_median(b_t_m_s)

        t_err_quant = np.quantile([np.linalg.norm(b_t_m[:3,3]-base_t_marker_median[:3,3]) for b_t_m in b_t_m_s], 1-base_t_gripper_outlier_quantiles[0])
        r_err_quant = np.quantile([calc_rotational_difference(b_t_m, base_t_marker_median) for b_t_m in b_t_m_s], 1-base_t_gripper_outlier_quantiles[1])

        for idx, (b_t_g, c_t_m) in enumerate(zip(base_t_gripper_s, camera_t_marker_s)):
            not_None:bool = c_t_m is not None
            low_t_err:bool = not_None and np.linalg.norm((b_t_g @ gripper_t_cam @ c_t_m)[:3,3] - base_t_marker_median[:3,3]) <= t_err_quant
            low_r_err:bool = not_None and calc_rotational_difference(b_t_g @ gripper_t_cam @ c_t_m, base_t_marker_median) <= r_err_quant

            if not low_t_err or not low_r_err:
                camera_t_marker_s[idx] = None

    return GatheredRobotData(
        name = name,
        bgr_images=np.array(rgb_images),
        color_cam_mtx=rgb_cam_mat,
        color_cam_distortion_coefficients=rgb_cam_dist_coef,
        depth_images=depth_images,
        depth_cam_mtx=depth_cam_mat,
        depth_cam_distortion_coefficients=depth_cam_distortion_coefficients,
        base_t_gripper_s=np.array(base_t_gripper_s),
        camera_t_marker_s=camera_t_marker_s,
        marker_detector=marker_detector,
        gripper_t_cam=gripper_t_cam
    )




def check_output_data(
        gathered_data: GatheredRobotData,
        output_folder:str = "data",
        use_mean:bool = True,
        save_result:bool = True,
):
    """
    Displays statistics for the estimated base_t_camera poses
    :param use_mean: Whether to treat the pose mean or median as the truth
    :param save_result: Whether to save the results in output_folder/data_collect_analysis.pdf or not
    :param output_folder: Where the result will be saved
    """
    base_t_marker_s = [r_t_g @ gathered_data.gripper_t_cam @ c_t_a for r_t_g, c_t_a in zip(base_t_gripper_s, gathered_data.camera_t_marker_s) if c_t_a is not None]

    if len(base_t_marker_s) < 1:
        return

    avg_base_t_marker = np.eye(4)
    if use_mean:
        from scipy.spatial.transform import RigidTransform
        avg_base_t_marker = RigidTransform.from_matrix(np.array(base_t_marker_s)).mean().as_matrix()
    median_base_t_marker = compute_pose_pseudo_median(np.array(base_t_marker_s))

    actual_base_t_marker = avg_base_t_marker if use_mean else median_base_t_marker


    translational_errors_mm = [np.linalg.norm(b_t_a[:3,3]-actual_base_t_marker[:3,3])*1000 for b_t_a in base_t_marker_s]
    rotational_errors_deg = [calc_rotational_difference(b_t_a[:3,:3], actual_base_t_marker[:3,:3])*360 for b_t_a in base_t_marker_s]

    # plot results:
    fig = plt.figure(figsize = (12, 6))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.4, height_ratios=[1, 1, 1, 2])

    # Basic stats:
    text_plt = fig.add_subplot(gs[0:3,0])
    text_plt.axis("off")
    text_plt.set_title(f"Gathered data analysis")

    cov_text = np.char.mod('%7.3f', np.round(np.cov(np.array(base_t_marker_s)[:,:3,3]*1000, rowvar = False), 3))
    cor_text = np.char.mod('%7.3f', np.round(np.corrcoef(np.array(base_t_marker_s)[:,:3,3]*1000, rowvar = False), 3))

    text = f"""
{len([p for p in gathered_data.camera_t_marker_s if p is not None])}/{len(gathered_data.camera_t_marker_s)} positions have marker pose estimates

Avg translational error: {np.round(np.mean(translational_errors_mm), 3)}mm
Avg rotational error: {np.round(np.mean(rotational_errors_deg), 3)}°
Reference Mat, decided by {"mean" if use_mean else "median" }:

{chr(10).join(["│"+"".join(row)+" │" for row in np.char.mod('%7.3f', np.round(actual_base_t_marker, 3))])}

Translation in mm cov & corr matrix:

{chr(10).join(["│"+"".join(row1)+" │  │"+" ".join(row2)+" │" for row1, row2 in zip(cov_text, cor_text)])}

    """
    text_plt.text(0.0, 1.0,text, verticalalignment = "top", horizontalalignment = "left", fontfamily='monospace')

    # Translational errors
    t_err_plt = fig.add_subplot(gs[3,0:2])
    t_err_plt.hist(translational_errors_mm, bins = np.arange(int(min(translational_errors_mm)), int(max(translational_errors_mm)))+1)
    t_err_plt.set_xlabel(f"Distance to {'mean' if use_mean else 'median' } in mm")
    t_err_plt.set_ylabel("Frequency")

    # Rotational errors
    r_err_plt = fig.add_subplot(gs[3,2:4])
    r_err_plt.hist(rotational_errors_deg, bins = np.arange(int(min(translational_errors_mm)), int(max(translational_errors_mm)+1)))
    r_err_plt.set_xlabel(f"Rotational distance to {'mean' if use_mean else 'median' } in degrees")
    r_err_plt.set_ylabel("Frequency")


    # plot estimated poses
    pos_3d_plot = fig.add_subplot(gs[0:3,2:4], projection='3d')

    np_base_t_marker_s = np.array(base_t_marker_s)

    for idx, color in enumerate(['red', 'green', 'blue']):
        pos_3d_plot.quiver(
            X = np_base_t_marker_s[:,0,3]*1000, Y = np_base_t_marker_s[:,1,3]*1000, Z = np_base_t_marker_s[:,2,3]*1000,
            U = np_base_t_marker_s[:,0,idx], V = np_base_t_marker_s[:,1,idx], W = np_base_t_marker_s[:,2,idx],
            color = color
        )
    pos_3d_plot.scatter(actual_base_t_marker[0,3]*1000,actual_base_t_marker[1,3]*1000,actual_base_t_marker[2,3]*1000,color = "red", s = 50)

    pos_3d_plot.set_xlim(np_base_t_marker_s[:,0,3].min()*1000, np_base_t_marker_s[:,0,3].max()*1000)
    pos_3d_plot.set_ylim(np_base_t_marker_s[:,1,3].min()*1000, np_base_t_marker_s[:,1,3].max()*1000)
    pos_3d_plot.set_zlim(np_base_t_marker_s[:,2,3].min()*1000, np_base_t_marker_s[:,2,3].max()*1000)

    pos_3d_plot.view_init(20,45)
    pos_3d_plot.set_xlabel("x-pos mm")
    pos_3d_plot.set_ylabel("y-pos mm")
    pos_3d_plot.set_zlabel("z-pos mm")

    if save_result:
        fig.savefig(f"{output_folder}/error_analysis.pdf")
    plt.show()


if __name__ == "__main__":
    dictionary_options = {
        "4X4_250": cv2.aruco.DICT_4X4_250,
        "5X5_100": cv2.aruco.DICT_5X5_100,
        "5X5_250": cv2.aruco.DICT_5X5_250,
        "6X6_250": cv2.aruco.DICT_6X6_250,
        "7X7_250": cv2.aruco.DICT_7X7_250,
        "7X7_1000": cv2.aruco.DICT_7X7_1000,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", type=str, default="my_data", help="Output Folder Location")
    parser.add_argument("--gripper-t-cam-path", type=str, default=None, help="Path to gripper_t_cam.npy file, if left to None will be estimated")

    parser.add_argument("--no-data-gathering", action = "store_false", help = "If used only optimization & evaluation may be done", dest = "gather_data")
    parser.add_argument("--max-number-positions", type=int, default=None, help="Maximum number of positions to gather data by the robot")

    parser.add_argument("--stabilisation-timeout", type=float, default=0.0, help="Timeout in seconds between robot moved to position and picture is taken")

    parser.add_argument("--marker-detection", type = str, default=None, help = "If Aruco / Charuco marker detection should be used, options: `None`(default), `Aruco`, `Charuco`")
    parser.add_argument("--aruco-marker-side-length", type=float, default=0.072, help="Aruco marker side lengths in meters")
    parser.add_argument("--aruco-marker-dictionary", type=str, default="6X6_250", help=f"Aruco dictionary to use, possible options are: {', '.join(dictionary_options.keys())}")
    parser.add_argument("--charuco-square-side-length", type=float, default=0.1, help="Charuco board square side length in meters")
    parser.add_argument("--charuco-board-size", type=int, default=[14,9], nargs=2, help="Size of the charuco board in squares")

    parser.add_argument("--pose-outlier-quants", type=float, default=[0.2,0.2], nargs=2, help="The quantiles of base_t_marker estimates to remove 1st arg: translation, 2nd arg: rotation")

    parser.add_argument("--no-result-analysation", action = "store_false", help = "If used there wont by any result analysation (pose deviation analysis)", dest = "analyze_results")
    parser.add_argument("--no-pdf-store", action = "store_false", help = "If used the analysis wont be stored into the dataset", dest = "save_analysis_results")
    parser.add_argument("--analysis-baseline", type=str, default="median", help = "Decides wheather to assume that the mean or the median is assumed to be the actual base_t_marker cam be `median`/`mean` in analysis")

    parser.set_defaults(gather_data = True, visualize_poses = True, analyze_results = True)
    args = parser.parse_args()

    print(f"Saving/loading data from: {os.path.abspath(args.output_folder)}")

    gripper_t_cam = None
    if args.gripper_t_cam_path is not None and os.path.exists(args.gripper_t_cam_path):
        gripper_t_cam = np.load(args.gripper_t_cam_path)
        print(f"Using precomputed cam_t_gripper: \n {np.round(gripper_t_cam, 3)} \n from {args.gripper_t_cam_path}")


    rgb_images, base_t_gripper_s, rgb_cam_mat, rgb_cam_dist_coef = None, None, None, None
    if args.gather_data:
        print("gathering data using the robot...")
        if os.path.exists(f"{args.output_folder}"):
            print(f"Output folder already exists, deleting it ...")
            shutil.rmtree(f"{args.output_folder}")

        from robot_interface import gather_robot_data
        rgb_images, base_t_gripper_s, rgb_cam_mat, rgb_cam_dist_coef = gather_robot_data(output_folder=args.output_folder, number_of_positions = args.max_number_positions)
    else:
        if not os.path.exists(f"{args.output_folder}"):
            raise FileNotFoundError(f"{args.output_folder} does not exist")
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

    folders = sorted(os.listdir(f"{args.output_folder}/robot"))
    depth_images = np.array([np.load(f"{args.output_folder}/robot/{folder}/depth.npy") for folder in folders])

    with open(f"{args.output_folder}/robot_cam_calibration.json", 'r') as f:
        json_file = json.load(f)
        depth_cam_mat = np.array(json_file["depth_camera_matrix"])
        depth_cam_dist_coef = json_file["depth_distortion_coefficients"]


    marker_detector = None

    if args.marker_detection is not None and args.marker_detection == "Aruco":
        marker_detector = ArucoDetector(
            aruco_marker_side_length=args.aruco_marker_side_length,
            aruco_marker_dictionary=args.aruco_marker_dictionary
        )
    if args.marker_detection is not None and args.marker_detection == "Charuco":
        marker_detector = CharucoDetector(
            board_size=(args.charuco_board_size[0], args.charuco_board_size[1]),
            square_size=args.charuco_square_side_length,
            marker_size=args.aruco_marker_side_length,
            aruco_dictionary=args.aruco_marker_dictionary
        )
    
    if marker_detector is not None:
        with open(f"{args.output_folder}/metadata.json", 'w') as f:
            json.dump(marker_detector.get_meta_data(), f, indent=4)
    else:
        with open(f"{args.output_folder}/metadata.json", 'w') as f:
            json.dump({"Aruco/Charuco Type":None}, f, indent=4)


    optimize_robot_data(
        bgr_images=rgb_images,
        base_t_gripper_s=base_t_gripper_s,
        color_cam_mat=rgb_cam_mat,
        color_cam_dist_coef=rgb_cam_dist_coef,
        depth_images=depth_images,
        depth_cam_mat=depth_cam_mat,
        depth_cam_distortion_coefficients=depth_cam_dist_coef,
        gripper_t_cam=gripper_t_cam,
        marker_detector = marker_detector,
        base_t_gripper_outlier_quantiles=(args.pose_outlier_quants[0], args.pose_outlier_quants[1]),
    ).save(folder=os.path.dirname(args.output_folder), new_name=os.path.basename(args.output_folder))

    gd = GatheredRobotData.from_folder(args.output_folder)

    if args.analyze_results:
        check_output_data(
            gathered_data=gd,
            output_folder = args.output_folder,
            use_mean= (True if args.analysis_baseline == "mean" else False),
            save_result=args.save_analysis_results
        )
