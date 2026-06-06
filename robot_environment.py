import argparse, time
from headset_data import *
from image_to_pointcloud import *


class RobotEnvironment:
    def __init__(
            self,
            name:str,
            robot_bgr_images:np.ndarray,
            robot_bgr_intrinsics:np.ndarray,
            robot_xyz_images:np.ndarray,
            robot_base_t_robot_camera_s:np.ndarray,
    ):
        """
        :param name: the name of the dataset (will be stored under it)
        :param robot_bgr_images: BGR images of the robot as a NxHxWx3-uint8 numpy array
        :param robot_bgr_intrinsics: Intrinsics BGR camera matrix of the robot (3x3 numpy array)
        :param robot_xyz_images: XYZ images from the pov of the robot as a NxHxWx3-float numpy array
        :param robot_base_t_robot_camera_s: The homogeneous robot_base->robot_camera transformation matrix as a Nx4x4-float numpy array
        """
        self._name = name

        assert assert_mxnx3_np_uint8_image_batch(robot_bgr_images)
        self._robot_bgr_images = robot_bgr_images

        assert assert_intrinsic_mat(robot_bgr_intrinsics, robot_bgr_images[0])
        self._robot_bgr_intrinsics = robot_bgr_intrinsics

        assert robot_xyz_images.shape == robot_bgr_images.shape
        assert np.issubdtype(robot_xyz_images.dtype, np.floating)
        self._robot_xyz_images = robot_xyz_images

        assert robot_base_t_robot_camera_s.shape == (robot_bgr_images.shape[0],4,4), f"Wrong shape of robot_base_t_robot_camera_s {robot_base_t_robot_camera_s.shape}"
        assert all([assert_homogeneous_mat(m) for m in robot_base_t_robot_camera_s])
        self._robot_base_t_robot_camera_s = robot_base_t_robot_camera_s


    @classmethod
    def from_folder(cls, load_folder:str):
        """
        Loads from a folder of the structure:
        `output_folder`
        ├── robot_cam_calibration.json
        ├── robot
        │   └── 00 to number of datapoints
        │       ├── A xyz.npy file with the world points associated to each pixel
        │       ├── A robot_base_t_robot_camera.json with the 4x4 transformation matrix between robot base and camera
        │       └── A rgb.png image with possible aruco markers digitally removed
        └── point_cloud.ply
        """

        robot_cam_cal = json.load(open(f"{load_folder}/robot_cam_calibration.json"))

        robot_folders = sorted([f"{folder}" for folder in os.listdir(f"{load_folder}/robot")])
        robot_folders = [f"{load_folder}/robot/{folder}" for folder in robot_folders]

        robot_bgr_images = np.array([cv2.imread(f"{folder}/rgb.png") for folder in robot_folders])
        robot_xyz_images = np.array([np.load(f"{folder}/xyz.npy") for folder in robot_folders])
        robot_base_t_robot_cam_s = np.array([json.load(open(f"{folder}/robot_base_t_robot_camera.json")) for folder in robot_folders])

        instance = cls(
            name = os.path.basename(load_folder),
            robot_bgr_images = robot_bgr_images,
            robot_bgr_intrinsics = np.array(robot_cam_cal["intrinsic_camera_matrix"]),
            robot_xyz_images=robot_xyz_images,
            robot_base_t_robot_camera_s=robot_base_t_robot_cam_s,
        )
        return instance
    
    @classmethod
    def from_gathered_robot_data(
        cls,
        robot_data:GatheredRobotData,
        number_of_sampled_datapoints: int = 10,
        only_sample_robot_datapoints_w_marker_estimates: bool = False,
        markers_use_advanced_removal: bool = False,
        est3d_xyz_image_gen_config:XYZImageGenerationConfig | None = XYZImageGenerationConfig(),
        est3d_xyz_icp_config:ICPAlignmentConfig | None = ICPAlignmentConfig(),
    )->'RobotEnvironment':
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
        assert number_of_sampled_datapoints > 0, f"Non positive number of datapoints: {number_of_sampled_datapoints} cant be sampled"

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
        robot_base_t_robot_camera_s = np.array([robot_base_t_robot_camera_s[i] for i in chosen_indices])

        if robot_data.marker_detector is not None:
            robot_bgr_images = robot_data.marker_detector.remove_markers(robot_bgr_images)

        # Generate 3D Point cloud
        print("Generating point cloud...")
        robot_bgr_images, robot_base_xyz_imgs, robot_cam_intrinsic_mtx = create_aligned_xyz_images(
            robot_base_t_robot_camera_s = robot_base_t_robot_camera_s,
            robot_bgr_images = np.array(robot_bgr_images),
            robot_depth_images = np.array(robot_depth_images),
            intrinsic_camera_matrix = robot_data.cam_intrinsic_mtx,
            image_gen_config = est3d_xyz_image_gen_config,
            icp_config = est3d_xyz_icp_config,
        )

        instance = cls(
            name = robot_data.name,
            robot_bgr_images = np.array(robot_bgr_images),
            robot_bgr_intrinsics = robot_cam_intrinsic_mtx,
            robot_xyz_images= np.array(robot_base_xyz_imgs),
            robot_base_t_robot_camera_s=robot_base_t_robot_camera_s,
        )
        return instance


    def save(self, folder_path:str, new_name:str|None=None):
        """
        Saves the instance to a folder, in the format that it can be recreated using `from_folder`.
        Will delete a folder if `folder_path/name` already exists.
        :param folder_path: The folder in which the instance should be saved
        :param new_name: The new name of the output_folder if not they will just use the name
        """
        location = f"{folder_path}/{new_name if new_name is not None else self.name}"

        if os.path.exists(location):
            print(f"Output folder already exists, deleting it ...")
            shutil.rmtree(location)
        os.makedirs(name=location, exist_ok=True)
        
        with open(f"{location}/robot_cam_calibration.json", 'w') as f:
            json.dump({'intrinsic_camera_matrix': self.robot_bgr_intrinsics.tolist()}, f, indent=4)

        padding = len(str(self.robot_bgr_images.shape[0]-1))
        for i in range(self.robot_bgr_images.shape[0]):
            robot_folder = f"{location}/robot/{str(i).zfill(padding)}"
            os.makedirs(robot_folder, exist_ok=True)

            cv2.imwrite(f"{robot_folder}/rgb.png", self.robot_bgr_images[i])
            np.save(f"{robot_folder}/xyz.npy", self.robot_xyz_images[i])

            with open(f"{robot_folder}/robot_base_t_robot_camera.json", 'w') as f:
                json.dump(self.robot_base_t_robot_camera_s[i].tolist(), f, indent=4)

    def visualize_3d_data(self, visualize:bool = True):
        """
        Visualizes the robot environment using open3d
        """
        import open3d as o3d
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(self.robot_xyz_images.reshape(-1,3))
        pcd.colors = o3d.utility.Vector3dVector(self.robot_bgr_images.reshape(-1,3).astype(np.float32)[:, ::-1]/255)

        base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)

        robot_camera_s = []
        for b_t_c in self.robot_base_t_robot_camera_s:
            cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            cam_frame.transform(b_t_c)
            robot_camera_s.append(cam_frame)
            robot_camera_s.append(create_3d_camera(
                base_t_camera=b_t_c,
                intrinsics=self.robot_bgr_intrinsics,
                hxw_img=self.robot_bgr_images[0],
                scale=0.1
            ))

        to_vis = [pcd, base_frame]+robot_camera_s
        if visualize:
            o3d.visualization.draw_geometries(to_vis, f"Robot environment: {self.name} visualization")
        return to_vis

    @property
    def name(self)->str:
        return self._name

    @property
    def robot_bgr_images(self)->np.ndarray:
        return self._robot_bgr_images

    @property
    def robot_bgr_intrinsics(self)->np.ndarray:
        return self._robot_bgr_intrinsics

    @property
    def robot_xyz_images(self)->np.ndarray:
        return self._robot_xyz_images

    @property
    def robot_base_t_robot_camera_s(self)->np.ndarray:
        return self._robot_base_t_robot_camera_s
    

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

    parser.add_argument(
        "--icp-alignment", type = str, default="no alginment", 
        help = f"How to do the icp alignment", choices=list(ICPAlignmentConfigs.keys()),
    )

    args = parser.parse_args()

    start_time = time.perf_counter()
    
    robot_data = GatheredRobotData.from_folder(args.robot_input_folder)
    robot_env = RobotEnvironment.from_gathered_robot_data(
        robot_data = robot_data,
        number_of_sampled_datapoints=args.number_of_sampled_datapoints,
        only_sample_robot_datapoints_w_marker_estimates = args.sample_only_w_aruco,
        markers_use_advanced_removal=args.use_advanced_marker_removal,
        est3d_xyz_image_gen_config = XYZImageGenerationConfig(), #TODO add args
        est3d_xyz_icp_config=ICPAlignmentConfigs[args.icp_alignment]

    )
    robot_env.save(os.path.dirname(args.robot_output_folder), new_name=os.path.basename(args.robot_output_folder))


    headset_data = HeadsetData.from_vrs_file(args.headset_vrs_file)
    headset_data = create_robot_bound_headset_data(headset_data, robot_data)
    headset_data.save(os.path.dirname(args.headset_output_folder), new_name=os.path.basename(args.headset_output_folder))

    rob_load = RobotEnvironment.from_folder(args.robot_output_folder)
    head_load = HeadsetData.from_folder(args.headset_output_folder)

    visualize_robot_camera_environment_combo(robot_env=rob_load, headset_rec=head_load)
    print(f"Data processing took {(time.perf_counter() - start_time):.6f} seconds")