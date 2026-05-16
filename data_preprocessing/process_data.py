import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_2_prediction import *
from gathering_2_preprocessing import GatheredRobotData, compute_pose_pseudo_median

from image_to_pointcloud import *
from load_and_save import *
import argparse


def process_data(
        robot_data:GatheredRobotData,
        headset_data:HeadsetData,
        number_of_sampled_datapoints: int = 10,
        only_sample_robot_datapoints_w_marker_estimates: bool = False,
        markers_use_advanced_removal: bool = False, #TODO (optional)
        est3d_use_map_anything: bool = True,
        est3d_use_sam3_for_foreground_seg: bool = True,
        est3d_custom_sam3_prompts: list[tuple[str, int]] | None = None,
        est3d_xyz_img_mapanything_crop_square:bool = False,
        est3d_xyz_img_upscaling:bool = False, #TODO (optional)
        est3d_xyz_img_camera_alginment_method:Literal["none", "simple", "kabsch-umeyama"] = "kabsch-umeyama",
        est3d_xyz_img_confidence_threshold: int = 10,
        est3d_point_cloud_foreground_masks_conf_threshold: float = 0.5,
        est3d_point_cloud_foreground_object_detection_threshold: float = 0.5,
        est3d_point_cloud_iforest_confidence_threshold: float = 0.0,
        est3d_use_depth_images: bool = True,
        est3d_debug_point_cloud_visualize_result: bool = False,
        est3d_debug_visualize_foreground_masks: bool = False,
    )->PredictionData:
    """
    :param robot_data: GatheredRobotData instance
    :param headset_data: HeadsetData instance
    :param number_of_sampled_datapoints: The number of datapoints in the resulting prediction data
    :param only_sample_robot_datapoints_w_marker_estimates: Will sample only images with markers -> might lead to less then `number_of_sampled_datapoints` datapoints.
    :param markers_use_advanced_removal: Not implemented yet
    :param est3d_use_map_anything: If yes map-anything will be used for xyz-image generation (recommended method), if not crude Depth-based methods
    :param est3d_use_sam3_for_foreground_seg: If sam3 should be used to have only foreground objects in the pointcloud
    :param est3d_xyz_img_mapanything_crop_square: Wheather the color-images should be cropped square before being passed into map-anything
    :param est3d_xyz_img_upscaling: Not implemented yet
    :param est3d_xyz_img_camera_alginment_method: The method to match the map-anything camera poses to the actual camera poses and transform the points accordingly
    :param est3d_xyz_img_confidence_threshold: The confidence threshhold for points for map-anything 
    :param est3d_point_cloud_foreground_masks_conf_threshold: The confidence threshhold for object detection by sam3
    :param est3d_point_cloud_foreground_object_detection_threshold: The confidence threshhold for the masks by sam3
    :param est3d_point_cloud_iforest_confidence_threshold: The expected contamination of the point cloud to be removed by iforest
    :param est3d_use_depth_images: If yes passes the depth images into mapanything (they should then have the same intrinsic mat as the color images)
    :param est3d_debug_point_cloud_visualize_result: Wheather or not to visualize the pointcloud using o3d
    :param est3d_debug_visualize_foreground_masks: Wheather or not to show the segmentation by sam3
    :return: PredictionData instance
    """

    # Check that the parameters are valid
    assert number_of_sampled_datapoints > 0, "Non positive number of datapoints cant be sampled"
    assert 0 <= est3d_xyz_img_confidence_threshold <= 100, "est3d_xyz_img_confidence_threshold out of range: 0-100"
    assert 0.0 <= est3d_point_cloud_foreground_object_detection_threshold <= 1.0, "est3d_pointcloud_foreground_object_detection_threshold out of range: 0.0-1.0"
    assert 0.0 <= est3d_point_cloud_foreground_masks_conf_threshold <= 1.0, "est3d_pointcloud_foreground_masks_conf_threshold out of range: 0.0-1.0"
    assert 0.0 <= est3d_point_cloud_iforest_confidence_threshold <= 1.0, "est3d_pointcloud_iforest_confidence_threshold out of range: 0.0-1.0"

    headset_images = headset_data.bgr_image_s
    headset_cam_mtx = headset_data.intrinsic_camera_matrix
    headset_cam_dist_coef = headset_data.distortion_coefficients

    robot_bgr_images = robot_data.bgr_images
    robot_depth_images = robot_data.depth_images
    robot_camera_t_marker_s = robot_data.camera_t_marker_s
    robot_base_t_robot_camera_s = robot_data.base_t_camera_s

    robot_bgr_cam_mtx = robot_data.color_cam_mtx
    robot_bgr_cam_dist_coef = robot_data.color_distortion_coefficients
    robot_depth_cam_mtx:np.ndarray = robot_data.depth_cam_mtx

    # Marker handling
    headset_image = headset_images[int(len(headset_images)/2)]
    headset_t_marker = None
    # select better headset image
    if robot_data.marker_detector is not None:
        headset_t_markers = robot_data.marker_detector.get_camera_t_marker(
            images=list(headset_images),
            camera_matrix=headset_cam_mtx,
            distortion_coefficients = robot_bgr_cam_dist_coef
        )
        headset_t_markers_idx_none_filtered = [idx for idx, h_t_m in enumerate(headset_t_markers) if h_t_m is not None]
        if len(headset_t_markers_idx_none_filtered) > 0:
            headset_w_marker_img_idx = min(headset_t_markers_idx_none_filtered, key = lambda x: np.abs(x-len(headset_images)/2))
            headset_t_marker = headset_t_markers[headset_w_marker_img_idx]
            headset_image = headset_images[headset_w_marker_img_idx]

        headset_image = robot_data.marker_detector.remove_markers([headset_image])[0]


    # Choose the robot images smartly
    robot_image_indices_w_base_t_marker = [idx for idx, c_t_m in enumerate(robot_camera_t_marker_s) if c_t_m is not None]

    chosen_indices = [idx for idx, _ in enumerate(robot_bgr_images)]

    if number_of_sampled_datapoints <= len(robot_image_indices_w_base_t_marker) or only_sample_robot_datapoints_w_marker_estimates:
        chosen_indices = robot_image_indices_w_base_t_marker[:number_of_sampled_datapoints]
    else:
        indices_no_marker_pose = set(chosen_indices) - set(robot_image_indices_w_base_t_marker)
        chosen_indices = robot_image_indices_w_base_t_marker + list(indices_no_marker_pose)[:number_of_sampled_datapoints-len(robot_image_indices_w_base_t_marker)]

    print(f"length of chosen indices {len(chosen_indices)}")

    robot_bgr_images = [robot_bgr_images[i] for i in chosen_indices]
    robot_depth_images = [robot_depth_images[i] for i in chosen_indices]
    robot_camera_t_marker_s = [robot_camera_t_marker_s[i] for i in chosen_indices]
    robot_base_t_robot_camera_s = [robot_base_t_robot_camera_s[i] for i in chosen_indices]

    if robot_data.marker_detector is not None:
        robot_bgr_images = robot_data.marker_detector.remove_markers(robot_bgr_images)

    # Generate 3D Point cloud
    print("Generating point cloud...")
    robot_base_xyz_imgs = None
    point_cloud = None
    image_mask_generator = lambda imgs: create_foreground_masks(
                images=imgs,
                threshold=est3d_point_cloud_foreground_object_detection_threshold,
                mask_threshold = est3d_point_cloud_foreground_masks_conf_threshold,
                visualize_masks=est3d_debug_visualize_foreground_masks,
                prompts=est3d_custom_sam3_prompts
    ) if est3d_use_sam3_for_foreground_seg else None

    if est3d_use_map_anything:
        robot_bgr_images, robot_base_xyz_imgs, point_cloud, robot_bgr_cam_mtx = create_point_cloud(
            bgr_images=np.array(robot_bgr_images),
            base_t_cam_s=np.array(robot_base_t_robot_camera_s),
            depth_images=np.array(robot_depth_images) if est3d_use_depth_images else None,
            camera_intrinsics=robot_bgr_cam_mtx,
            confidence_threshold_percent=est3d_xyz_img_confidence_threshold,
            image_mask_generator=image_mask_generator,
            visualize_point_cloud=est3d_debug_point_cloud_visualize_result,
            alginment_method="kabsch-umeyama",
            crop_square=est3d_xyz_img_mapanything_crop_square
        )
        # Use Iforest on pointcloud
        point_cloud = remove_outliers_from_point_cloud(
            points=point_cloud,
            contamination=est3d_point_cloud_iforest_confidence_threshold
        )
    else:
        robot_base_xyz_imgs, point_cloud = create_point_cloud_simple(
            depth_images=np.array(robot_depth_images),
            depth_cam_mtx=np.array(robot_depth_cam_mtx),
            base_t_camera_s=np.array(robot_base_t_robot_camera_s),
            image_masks=image_mask_generator(np.array(robot_bgr_images)) if image_mask_generator else None,
            distance_cutoff=1.0,
            visualize_point_cloud=True
        )

    print("Generating the label...")
    robot_base_t_headsets = [None] * len(robot_bgr_images)
    if headset_t_marker is not None:
        for idx, (robot_base_t_camera, robot_camera_t_marker) in enumerate(zip(robot_base_t_robot_camera_s, robot_camera_t_marker_s)):
            if robot_camera_t_marker is not None:
                robot_base_t_headsets[idx] = robot_base_t_camera @ robot_camera_t_marker @ np.linalg.inv(headset_t_marker)
    robot_base_t_headset = compute_pose_pseudo_median([m for m in robot_base_t_headsets if m is not None])

    return PredictionData(
        name = "",
        robot_bgr_images=np.array(robot_bgr_images),
        robot_bgr_intrinsics=robot_bgr_cam_mtx,
        headset_bgr_image=headset_image,
        headset_intrinsics=headset_cam_mtx,
        robot_xyz_images=np.array(robot_base_xyz_imgs),
        point_cloud=point_cloud,
        robot_base_t_robot_camera_s=np.array(robot_base_t_robot_camera_s),
        robot_base_t_headset=robot_base_t_headset
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-input-folder", type=str, default="./in_folder", help="Robot input Folder Location")
    parser.add_argument("--headset-vrs-file", type=str, default="", help=".vrs file location")
    parser.add_argument("--output-folder", type=str, default="./out_data", help="Output Folder Location")
    parser.add_argument("--number-of-sampled-datapoints", type=int, default=9999, help="Max number of input points to be sampled")
    parser.add_argument("--dont-limit-to-only-aruco", action="store_false", dest="sample_only_w_aruco")
    parser.add_argument("--dont-use-map-anything", action="store_false", dest="use_map_anything")
    parser.add_argument("--dont-use-sam3", action="store_false", dest="use_sam3")
    parser.add_argument("--dont-crop-to-square", action="store_false", dest="crop_square")
    parser.add_argument("--camera-alignment-method", type=str, default="kabsch-umeyama", help="What algorithm to use to align mapanything and real world cameras: none, simple, kabsch-umeyama")
    parser.add_argument("--mapanything-point-conf-threshhold", type=int, default=10)
    parser.add_argument("--sam3-mask-threshhold", type=float, default=0.5)
    parser.add_argument("--sam3-object-threshhold", type=float, default=0.5)
    parser.add_argument("--iforest-contamination", type = float, default=0.05)
    parser.add_argument("--dont-use-depth-images", action="store_false", dest="use_depth_images")

    args = parser.parse_args()

    start_time = time.perf_counter()
    
    robot_data = GatheredRobotData.from_folder(args.robot_input_folder)
    headset_data = HeadsetData.from_vrs_file(args.headset_vrs_file)

    processed_data = process_data(
        robot_data = robot_data,
        headset_data=headset_data,
        number_of_sampled_datapoints=args.number_of_sampled_datapoints,
        only_sample_robot_datapoints_w_marker_estimates=args.sample_only_w_aruco,
        est3d_use_map_anything=args.use_map_anything,
        est3d_use_sam3_for_foreground_seg=args.use_sam3,
        est3d_xyz_img_mapanything_crop_square=args.crop_square,
        est3d_xyz_img_camera_alginment_method= args.camera_alignment_method,
        est3d_xyz_img_confidence_threshold=args.mapanything_point_conf_threshhold,
        est3d_point_cloud_foreground_masks_conf_threshold= args.sam3_mask_threshhold,
        est3d_point_cloud_foreground_object_detection_threshold = args.sam3_object_threshhold,
        est3d_point_cloud_iforest_confidence_threshold = args.iforest_contamination,
        est3d_use_depth_images = args.use_depth_images,
    )
    processed_data.save(os.path.dirname(args.output_folder), new_name=os.path.basename(args.output_folder))
    pd = PredictionData.from_folder(args.output_folder)
    pd.visualize_3d_data()
    print(f"Data processing took {(time.perf_counter() - start_time):.6f} seconds")