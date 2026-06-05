
Proto-Robot-Data:
    -depth_images:np.ndarray,
    -cam_intrinsic_mtx:np.ndarray,
    -cam_distortion_coefficients:list[float],
    -bgr_images:np.ndarray,
    -base_t_gripper_s:np.ndarray,

Gathered-Robot-Data:
    name:str,
    bgr_images:np.ndarray,
    cam_intrinsic_mtx:np.ndarray,
    depth_images:np.ndarray,
    base_t_gripper_s:np.ndarray,
    camera_t_marker_s:list[None | np.ndarray],
    marker_detector: MarkerDetector,
    gripper_t_cam: np.ndarray,


Robot-Environment:
    name:str,
    robot_bgr_images:np.ndarray,
    robot_bgr_intrinsics:np.ndarray,
    robot_xyz_images:np.ndarray,
    robot_base_t_robot_camera_s:np.ndarray,


headset_data:
    name:str,
    bgr_image_s: np.ndarray,
    intrinsic_cam_mtx: np.ndarray,
    robot_base_t_headset_s: list[np.ndarray | None]


optimize_robot_data(Proto-Robot-Data, Marker) -> Gathered-Robot-Data

gathered_robot_data -> robot-environment

.vrs -> headset_data
headset_data + gathered_robot_data -> annotated[headset_data]


