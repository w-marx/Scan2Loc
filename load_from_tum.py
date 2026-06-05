import os, cv2
import numpy as np
import pandas as pd
from robot_environment import RobotEnvironment
from headset_data import HeadsetData
from shared_utilities import *
from image_to_pointcloud import create_aligned_xyz_images, ICPAlignmentConfig, XYZImageGenerationConfig
from typing import Literal


def load_bgr_images(folder:str)->tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(
        f"{folder}/rgb.txt", comment = '#', delim_whitespace=True, names=["timestamp", "location"]
    )
    df = df.sort_values('timestamp')
    df['image'] = df['location'].apply(
        lambda x: cv2.imread(f"{folder}/{x}") if os.path.exists(f"{folder}/{x}") else None
    )
    df = df.dropna(subset=['image']).copy()

    timestamps = df['timestamp'].values
    bgr_images = df['image'].values

    assert assert_mxnx3_np_uint8_image_batch

    return timestamps, bgr_images

def load_depth_images(folder:str)->tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(
        f"{folder}/depth.txt", comment = '#', delim_whitespace=True, names=["timestamp", "location"]
    )
    df = df.sort_values('timestamp')
    df['image'] = df['location'].apply(
        lambda x: cv2.imread(str(f"{folder}/{x}"), cv2.IMREAD_UNCHANGED) if os.path.exists(f"{folder}/{x}") else None
    )
    df = df.dropna(subset=['image']).copy()

    timestamps = df['timestamp'].values
    depth_images_unnorm = df['image'].values
    depth_images = depth_images_unnorm/5000

    assert assert_mxn_np_float_image_batch(depth_images)

    return timestamps, depth_images



def load_labels(file:str)->tuple[np.ndarray, np.ndarray]:
    """
    Load poses and timestamps from .txt
    :param file: The file location
    :return: float-array of timestamps + 4x4 hom. pose matrices
    """
    df = pd.read_csv(
        file, comment = '#', delim_whitespace=True, names=["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
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

    timestamps_bgr, bgr_images = load_bgr_images(folder=folder)
    timestamps_depth, depth_images = load_depth_images(folder=folder)
    timestamps_world_t_c, world_t_cam_s = load_labels(f"{folder}/groundtruth.txt")
    
    synchronized_timestamps = synchronize_timestamps([timestamps_bgr, timestamps_depth, timestamps_world_t_c], tolerance=time_tolerance)

    n_dp = synchronized_timestamps.shape[0]
    sync_bgr_images = bgr_images[synchronize_timestamps[:,0]]
    sync_depth_images = depth_images[synchronize_timestamps[:,1]]
    sync_world_t_cam_s = world_t_cam_s[synchronize_timestamps[:,2]]


    # Robot environment generation
    robot_indices = range(0, n_dp, step = int(n_dp/n_robot_images))

    robot_bgr_images, robot_xyz_images, robot_intrinsics = create_aligned_xyz_images(
        robot_bgr_images=sync_bgr_images[robot_indices],
        robot_depth_images=sync_depth_images[robot_indices],
        intrinsic_camera_matrix=INTRINSIC_FREIBURG_MATRICES[rgb_camera_name],
        image_gen_config=xyz_image_generation_config,
        icp_config=xyz_image_alginment_config
    )

    robot_env = RobotEnvironment(
        name=f"{os.path.basename(folder)}_robot_env",
        robot_bgr_images=robot_bgr_images,
        robot_bgr_intrinsics=robot_intrinsics,
        robot_xyz_images=robot_xyz_images,
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