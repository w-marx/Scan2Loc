import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from robot_environment import *
from gathered_robot_data import GatheredRobotData
from headset_data import *

from image_to_pointcloud import *
#from data_preprocessing.headset_data_from_vrs import *
import argparse



def process_robot_data(
        robot_data:GatheredRobotData,
        number_of_sampled_datapoints: int = 10,
        only_sample_robot_datapoints_w_marker_estimates: bool = False,
        markers_use_advanced_removal: bool = False,
        est3d_xyz_image_gen_config:XYZImageGenerationConfig | None = XYZImageGenerationConfig(),
        est3d_xyz_icp_config:ICPAlignmentConfig | None = ICPAlignmentConfig(),
    )->RobotEnvironment:
    """
    :param robot_data: GatheredRobotData instance
    :param number_of_sampled_datapoints: The number of datapoints in the resulting prediction data
    :param only_sample_robot_datapoints_w_marker_estimates: Will sample only images with markers -> might lead to less then `number_of_sampled_datapoints` datapoints.
    :param markers_use_advanced_removal: If yes will use an ai-image inpainting tool and not just replace the marker with a black blob
    :param est3d_xyz_image_gen_config: If not None will be used for xyz-image generation via mapanything (recommended method), if not crude Depth-based methods
    :param est3d_xyz_icp_config: Do icp alignment of the xyz-images using the config if not None
    :return: RobotEnvironment instance
    """

    # Check that the parameters are valid
    assert number_of_sampled_datapoints > 0, "Non positive number of datapoints cant be sampled"

    robot_bgr_images = robot_data.bgr_images
    robot_depth_images = robot_data.depth_images
    robot_camera_t_marker_s = robot_data.camera_t_marker_s
    robot_base_t_robot_camera_s = robot_data.base_t_camera_s

    if robot_data.marker_detector is not None:
        robot_data.marker_detector.set_new_masker("LamaMasker" if markers_use_advanced_removal else "ImageMasker")


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

    if est3d_xyz_image_gen_config is not None:
        robot_bgr_images, robot_base_xyz_imgs, robot_cam_intrinsic_mtx = generate_xyz_images(
            bgr_images=np.array(robot_bgr_images),
            base_t_cam_s=np.array(robot_base_t_robot_camera_s),
            depth_images=np.array(robot_depth_images),
            camera_intrinsics=robot_data.cam_intrinsic_mtx,
            config=est3d_xyz_image_gen_config
        )
    else:
        robot_base_xyz_imgs = create_point_cloud_depth_reproject(
            depth_images=np.array(robot_depth_images),
            depth_cam_mtx=robot_data.cam_intrinsic_mtx,
            base_t_camera_s=np.array(robot_base_t_robot_camera_s),
            distance_cutoff=1.0,
            visualize_point_cloud=True
        )
    
    if est3d_xyz_icp_config is not None:
        robot_base_xyz_imgs = [
            xyz_img.reshape(robot_base_xyz_imgs.shape[1:]) for xyz_img in 
            align_point_clouds_icp(
                point_clouds = [xyz_img.reshape(-1,3) for xyz_img in robot_base_xyz_imgs],
                config = est3d_xyz_icp_config,
            )
        ]

    return RobotEnvironment(
        name = "",
        robot_bgr_images=np.array(robot_bgr_images),
        robot_bgr_intrinsics=robot_cam_intrinsic_mtx,
        robot_xyz_images=np.array(robot_base_xyz_imgs),
        robot_base_t_robot_camera_s=np.array(robot_base_t_robot_camera_s),
    )


def visualize_robot_camera_environment_combo(robot_env:RobotEnvironment, headset_rec:HeadsetData):
    to_vis_robot = robot_env.visualize_3d_data(visualize=False)
    to_vis_headset = headset_rec.visualize_3d_data(visualize=False)
    o3d.visualization.draw_geometries(
        to_vis_robot+to_vis_headset, 
        f"Robot: {robot_env.name} x Headset: {headset_rec.name} visualization"
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--robot-input-folder", type=str, default="./in_folder", help="Robot input Folder Location")
    parser.add_argument("--headset-vrs-file", type=str, default="", help=".vrs file location")
    parser.add_argument("--robot-output-folder", type=str, default="./out_data_r", help="Output Folder Location for the Robot environment")
    parser.add_argument("--headset-output-folder", type=str, default="./out_data_h", help="Output Folder Location for the headset recording")


    parser.add_argument("--number-of-sampled-datapoints", type=int, default=9999, help="Max number of input points to be sampled")
    parser.add_argument("--dont-limit-to-only-aruco", action="store_false", dest="sample_only_w_aruco")
    parser.add_argument("--dont-use-ai-marker-removal", action="store_false", dest="use_advanced_marker_removal")

    args = parser.parse_args()

    start_time = time.perf_counter()
    
    robot_data = GatheredRobotData.from_folder(args.robot_input_folder)
    processed_robot_data = process_robot_data(
        robot_data = robot_data,
        number_of_sampled_datapoints=args.number_of_sampled_datapoints,
        only_sample_robot_datapoints_w_marker_estimates = args.sample_only_w_aruco,
        markers_use_advanced_removal=args.use_advanced_marker_removal,
        est3d_xyz_image_gen_config = XYZImageGenerationConfig(), #TODO add args
        est3d_xyz_icp_config=None#ICPAlignmentConfigs["downsample_5mm"]

    )
    processed_robot_data.save(os.path.dirname(args.robot_output_folder), new_name=os.path.basename(args.robot_output_folder))


    headset_data = HeadsetData.from_vrs_file(args.headset_vrs_file)
    headset_data = create_robot_bound_headset_data(headset_data, robot_data)
    headset_data.save(os.path.dirname(args.headset_output_folder), new_name=os.path.basename(args.headset_output_folder))



    rob_load = RobotEnvironment.from_folder(args.robot_output_folder)
    head_load = HeadsetData.from_folder(args.headset_output_folder)

    visualize_robot_camera_environment_combo(robot_env=rob_load, headset_rec=head_load)
    print(f"Data processing took {(time.perf_counter() - start_time):.6f} seconds")