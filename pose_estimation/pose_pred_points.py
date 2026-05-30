import cv2
import numpy as np
from predictor_handling import *
from extractors_and_matchers import *
from image_augmentation import *
from sheduler import *

class NoExtrasPredictor(PosePredictor):
    def __init__(
            self,
            cam2_mtx:np.ndarray,
            cam1_bgr_images:np.ndarray,
            cam1_xyz_images:np.ndarray,
            extract_and_match:ExtractAndMatch = ExtractAndMatchLoMa(),
            use_rotation_augmentations:bool = False,
            crop_augmentations:list[float] | None = None,
            ransac_config:RansacPoseEstimationConfig = pose_estimation_ransaac_config_precise,
            sheduler:Sheduler = EMASheduler,
            number_tries_b4_giving_up:int = 1,
        ):
        super().__init__()
        self.cam2_mtx = cam2_mtx
        self.extract_and_match = extract_and_match
        self.ransac_config = ransac_config

        self.rotation_augmentations = [Augmentation()]
        if use_rotation_augmentations:
            self.rotation_augmentations.append(Rotate180Deg())
        
        self.crop_augmentations = [Augmentation()]
        if crop_augmentations is not None:
            self.crop_augmentations + [CropImage(x) for x in crop_augmentations if 0 <= x < 1.0]  

        self.use_rotation_augmentations = use_rotation_augmentations
        self.number_tries_b4_giving_up = number_tries_b4_giving_up

        self.cam1_bgr_images = cam1_bgr_images
        self.cam1_xyz_images = cam1_xyz_images

        self.sheduler = sheduler(cam1_bgr_images.shape[0])
    
    def est_base_t_cam2(self,cam2_bgr_image: np.ndarray,time_tracker:TimeTracker) -> np.ndarray | None:
        number_tries = 0
        est_base_t_cam = None
        while est_base_t_cam is None and number_tries < self.number_tries_b4_giving_up:
            idx = self.sheduler.get_best()
            print(f"idx: {idx}")
            est_base_t_cam = self.est_base_t_cam2_helper(
                cam1_bgr_image = self.cam1_bgr_images[idx],
                base_xyz_image = self.cam1_xyz_images[idx],
                cam2_bgr_image = cam2_bgr_image,
                time_tracker = time_tracker
            )
            self.sheduler.adjust(idx, est_base_t_cam is not None)
            number_tries += 1
        return est_base_t_cam
    
    def update_pose(self,cam2_bgr_image: np.ndarray, old_pose:np.ndarray, time_tracker:TimeTracker) -> np.ndarray | None:
        return self.est_base_t_cam2(cam2_bgr_image=cam2_bgr_image, time_tracker=time_tracker)


    def est_base_t_cam2_helper(self,
                        cam1_bgr_image:np.ndarray,
                        base_xyz_image:np.ndarray,
                        cam2_bgr_image: np.ndarray,
                        time_tracker:TimeTracker
                        ) -> np.ndarray | None:
        
        time_tracker.reset_elapsed_time()
        
        augmentation_options_names = []
        world_obj_points_options = []
        image_points_cam2_options = []

        for c_aug in self.crop_augmentations:
            for r_aug in self.rotation_augmentations:
                augmented_image, backward_aug2 = c_aug.forward(cam2_bgr_image)
                augmented_image, backward_aug1 = r_aug.forward(augmented_image)

                time_tracker.add_time_stamp("image augmentation")

                image_points_cam1, image_points_cam2 = self.extract_and_match.get_matched_points(
                    img1_rgb=cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2RGB),
                    img2_rgb=cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                )

                time_tracker.add_time_stamp("extract and match")

                world_obj_points = np.array([base_xyz_image[int(np.round(y)),int(np.round(x))] for x,y in image_points_cam1])
                if world_obj_points.shape[0] > 5:
                    augmentation_options_names.append(f"{c_aug}x{r_aug}")
                    world_obj_points_options.append(world_obj_points)
                    image_points_cam2_options.append(backward_aug2(backward_aug1(image_points_cam2)))

                time_tracker.add_time_stamp("world point extraction")
        
        if len(world_obj_points_options) < 1:
            return None
    
        best_option_idx = np.argmax(np.array([x.shape[0] for x in world_obj_points_options]))
        world_obj_points = world_obj_points_options[best_option_idx]
        image_points_cam2 = image_points_cam2_options[best_option_idx]

        time_tracker.reset_elapsed_time()
        cam2_t_base__inliers = estimate_point_pose_ransac(
            world_points=world_obj_points,
            img_points=image_points_cam2,
            intrinsic_matrix=self.cam2_mtx,
            config=self.ransac_config
        )
        if cam2_t_base__inliers is None:
            return None
        return np.linalg.inv(cam2_t_base__inliers[0])




if __name__ == "__main__":
    robot_data = RobotEnvironment.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data_r")
    headset_data = HeadsetData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data_h")
    predictor = NoExtrasPredictor(
        cam2_mtx=headset_data.intrinsic_cam_mtx,
        cam1_bgr_images=robot_data.robot_bgr_images,
        cam1_xyz_images=robot_data.robot_xyz_images,
        extract_and_match=ExtractAndLightGlue(),
        use_rotation_augmentations=False,
        number_tries_b4_giving_up=10
    )

    tt1 = TimeTracker()
    tt2 = TimeTracker()
    grader = OnePredictorRecordingGrader(
        predictor=predictor, 
        headset_data=headset_data,
        prediction_time_tracker=tt1,
        subcomponent_time_tracker=tt2
    )
    
    print(f"translat errors: \n {grader.translational_errors()} \n")
    #grader.visualize_predictions()
    print(f"avg rot error: {np.round(np.rad2deg(grader.avg_rotational_error()), 2)} degrees")
    print(f"avg translational error: {np.round(grader.avg_translational_error()*1000, 1)} mm")
    print(f"median rot error: {np.round(np.rad2deg(grader.median_rotational_error()), 2)} degrees")
    print(f"median translational error: {np.round(grader.median_translational_error()*1000, 1)} mm")
    print(f"sucess_ratio: {np.round(grader.success_ratio(),2)}")

    grader.visualize_predictions(robot_env=robot_data)

    print(f"tt1:")
    tt1.print_report()
    print(f"\n tt2:")
    tt2.print_report()