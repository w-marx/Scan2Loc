import os, cv2
import numpy as np
import pandas as pd
from robot_environment import RobotEnvironment, visualize_robot_camera_environment_combo
from headset_data import HeadsetData
from shared_utilities import *
from image_to_pointcloud import create_aligned_xyz_images, ICPAlignmentConfig, XYZImageGenerationConfig, ICPAlignmentConfigs, XYZImageGenerationConfigs
from typing import Literal
import argparse


def load_bgr_images(folder:str)->tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(
        f"{folder}/rgb.txt", comment = '#', names=["timestamp", "location"], sep=r"\s+"
    )
    df['image'] = df['location'].apply(
        lambda x: cv2.imread(f"{folder}/{x}") if os.path.exists(f"{folder}/{x}") else None
    )
    df = df.dropna(subset=['image']).copy()

    timestamps = np.array(df['timestamp'].values)
    bgr_images = np.stack(df['image'].values, axis = 0)

    assert assert_mxnx3_np_uint8_image_batch(bgr_images)

    return timestamps, bgr_images

def load_depth_images(folder:str)->tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(
        f"{folder}/depth.txt", comment = '#', names=["timestamp", "location"], sep=r"\s+"
    )
    df = df.sort_values('timestamp')
    df['image'] = df['location'].apply(
        lambda x: cv2.imread(str(f"{folder}/{x}"), cv2.IMREAD_UNCHANGED) if os.path.exists(f"{folder}/{x}") else None
    )
    df = df.dropna(subset=['image']).copy()

    timestamps = np.array(df['timestamp'].values)
    depth_images_unnorm = np.stack(df['image'].values, axis = 0)

    depth_images = depth_images_unnorm/5000
    depth_images[depth_images_unnorm == 0] = np.nan

    assert assert_mxn_np_float_image_batch(depth_images)

    return timestamps, depth_images



def load_labels(file:str)->tuple[np.ndarray, np.ndarray]:
    """
    Load poses and timestamps from .txt
    :param file: The file location
    :return: float-array of timestamps + 4x4 hom. pose matrices
    """
    df = pd.read_csv(
        file, comment = '#', names=["timestamp","tx", "ty", "tz", "qx", "qy", "qz", "qw"], sep=r"\s+"
    )
    df = df.sort_values('timestamp')

    base_t_cam_s = []
    for _, row in df.iterrows():
        base_t_cam_s.append(
            t_quat_to_hom(
                t = np.array([row['tx'], row['ty'], row['tz']]),
                quat=[row['qx'], row['qy'], row['qz'], row['qw']]
            )
        )
    base_t_cam_s = np.array(base_t_cam_s)
    assert assert_homogeneous_mat_batch(base_t_cam_s)

    timestamps = df['timestamp'].values
    return timestamps, base_t_cam_s


def synchronize_timestamps(
        timestamps:list[np.ndarray], 
        tolerance:float = 0.01
    )-> np.ndarray:
    """
    Takes B timestamp arrays (timestamps in global time) and returns B indice arrays to select the syncronisation
    :param timestamps: List of [sorted float timestamp array of length Ni]
    :return: A MxN matrix of indices such that timestamps[i][indices[i]] ~ timestamps[j][indices[j]]
    """

    timestamp_lengths = np.array([len(arr) for arr in timestamps])

    indices = []

    loop_indices = np.zeros(len(timestamps), dtype = np.int64)
    while np.all(loop_indices < timestamp_lengths):
        c_times = np.array([timestamps[i][loop_indices[i]] for i in range(len(timestamps))])

        t_min = np.min(c_times)
        t_max = np.max(c_times)

        if t_max - t_min <= tolerance:
            indices.append(loop_indices.copy())
            loop_indices += 1
            continue

        loop_indices[np.argmin(c_times)] += 1

    return np.array(indices)



INTRINSIC_FREIBURG_MATRICES = {
    "freiburg1" : build_intrinsic_mat(fx=517.3, fy = 516.5, cx = 318.6, cy = 255.3),
    "freiburg2" : build_intrinsic_mat(fx = 520.9, fy =512.0, cx = 325.1, cy = 249.7),
    "freiburg3" : build_intrinsic_mat(fx = 535.4, fy = 539.2, cx = 320.1, cy = 247.6)
}
    


def robot_environment_and_headset_data_from_tum(
        folder:str,
        rgb_camera_name:Literal["freiburg1", "freiburg2", "freiburg3"],
        time_tolerance:float = 0.1,
        n_robot_images:int = 10,
        xyz_image_generation_config:XYZImageGenerationConfig = XYZImageGenerationConfig(crop_square=False),
        xyz_image_alginment_config:ICPAlignmentConfig = ICPAlignmentConfig()
)-> tuple[RobotEnvironment, HeadsetData]:
    assert n_robot_images > 0

    if not os.path.exists(folder):
        raise FileNotFoundError(f"folder: {folder} doesnt exist")

    timestamps_bgr, bgr_images = load_bgr_images(folder=folder)
    timestamps_depth, depth_images = load_depth_images(folder=folder)
    timestamps_world_t_c, world_t_cam_s = load_labels(f"{folder}/groundtruth.txt")
    
    synchronized_timestamps = synchronize_timestamps([timestamps_bgr, timestamps_depth, timestamps_world_t_c], tolerance=time_tolerance)

    n_dp = synchronized_timestamps.shape[0]
    print(f"was able to match: {n_dp}/ ({timestamps_bgr.shape[0]}, {timestamps_depth.shape[0]}, {timestamps_world_t_c.shape[0]}) timestamps")

    sync_bgr_images = bgr_images[synchronized_timestamps[:,0]]
    sync_depth_images = depth_images[synchronized_timestamps[:,1]]
    sync_world_t_cam_s = world_t_cam_s[synchronized_timestamps[:,2]]


    # Robot environment generation
    robot_indices = range(0, n_dp, int(n_dp/n_robot_images))

    robot_bgr_images, robot_xyz_images, robot_intrinsics = create_aligned_xyz_images(
        robot_base_t_robot_camera_s=sync_world_t_cam_s[robot_indices],
        robot_bgr_images=sync_bgr_images[robot_indices],
        robot_depth_images=sync_depth_images[robot_indices],
        intrinsic_camera_matrix=INTRINSIC_FREIBURG_MATRICES[rgb_camera_name],
        image_gen_config=xyz_image_generation_config,
        icp_config=xyz_image_alginment_config
    )

    robot_env = RobotEnvironment(
        name=f"{os.path.basename(folder)}_robot_env",
        robot_bgr_images=np.array(robot_bgr_images),
        robot_bgr_intrinsics=robot_intrinsics,
        robot_xyz_images=np.array(robot_xyz_images),
        robot_base_t_robot_camera_s=sync_world_t_cam_s[robot_indices]
    )

    # Headset data generation
    headset_data = HeadsetData(
        name=f"{os.path.basename(folder)}_headset_data",
        bgr_image_s=sync_bgr_images,
        intrinsic_cam_mtx=INTRINSIC_FREIBURG_MATRICES[rgb_camera_name],
        robot_base_t_headset_s=list(sync_world_t_cam_s)
    )

    return robot_env, headset_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", type=str, default="./in_folder", help="Robot input Folder Location")

    parser.add_argument("--number-of-robot-datapoints", type=int, default=30, help="Number of datapoints that are used for the robot environment")
    parser.add_argument("--time-tolerance", type=float, default=0.05, help="Max difference between 2 datapoint timestamps to be matched")

    parser.add_argument(
        "--rgb-camera-name", type = str, default="freiburg2", 
        help = f"What camera generated the data", choices=list(INTRINSIC_FREIBURG_MATRICES.keys()),
    )
    parser.add_argument(
        "--icp-alignment", type = str, default="no alginment", 
        help = f"How to do the icp alignment", choices=list(ICPAlignmentConfigs.keys()),
    )
    parser.add_argument(
        "--xyz-image-gen", type = str, default="standard", 
        help = f"How to do the xyz image generation", choices=list(XYZImageGenerationConfigs.keys()),
    )

    parser.add_argument("--output-base-folder", type=str, default="./out_data", help="Output Folder Location")

    args = parser.parse_args()
    
    robot_env, headset_data = robot_environment_and_headset_data_from_tum(
        folder=args.input_folder,
        rgb_camera_name=args.rgb_camera_name,
        time_tolerance= args.time_tolerance,
        n_robot_images=args.number_of_robot_datapoints,
        xyz_image_generation_config=XYZImageGenerationConfigs[args.xyz_image_gen],
        xyz_image_alginment_config=ICPAlignmentConfigs[args.icp_alignment],
    )

    robot_env.save(args.output_base_folder)
    headset_data.save(args.output_base_folder)

    rob_load = RobotEnvironment.from_folder(f"{args.output_base_folder}/{os.path.basename(args.input_folder)}_robot_env")
    head_load = HeadsetData.from_folder(f"{args.output_base_folder}/{os.path.basename(args.input_folder)}_headset_data")

    visualize_robot_camera_environment_combo(robot_env=rob_load, headset_rec=head_load)