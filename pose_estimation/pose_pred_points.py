import cv2
import numpy as np
from predictor_handling import *
from extractors_and_matchers import *
from shared.assertion_helpers import assert_intrinsic_mat

from shared.src.shared.assertion_helpers import assert_mxn_np_float_image_batch, assert_mxnx3_np_uint8_image_batch
from small_utilities.image_augmentation import *

class OnlyPointsPredictor(PosePredictor):
    def __init__(
            self,
            cam2_intrinsic_mtx:np.ndarray,
            cam1_bgr_images:np.ndarray,
            cam1_xyz_images:np.ndarray,
            extract_and_match_wrapper_config:ExtractAndMatchWrapperConfig,
            time_tracker_init:TimeTracker = TimeTracker()
        ):
        """
        Initialises an predictor that only uses points as features.
        :param cam2_intrinsic_mtx: The 3x3 intrinsic matrix for camera 2
        :param cam1_bgr_images: BxHxWx3-uint8 array of bgr images for camera 1
        :param cam1_xyz_images: BxHxWx3-float array of xyz-point images for camera 1 in the base ref. frame
        :param extract_and_match_wrapper_config: The configuration for how to extract and match the points
        :param time_tracker_init: A timetracker where important steps during the initialization will be registered
        """
        super().__init__()
        assert assert_intrinsic_mat(cam2_intrinsic_mtx)
        assert assert_mxnx3_np_uint8_image_batch(cam1_bgr_images)
        assert assert_mxnx3_np_float_image_batch(cam1_xyz_images)
        assert cam1_bgr_images.shape[0] > 0, f"cant predict on no data"
        assert cam1_xyz_images.shape == cam1_xyz_images.shape

        time_tracker_init.reset_elapsed_time()
        self.extract_and_match_wrapper = ExtractAndMatchWrapper(
            cam2_mtx=cam2_intrinsic_mtx,
            cam1_bgr_images=cam1_bgr_images,
            cam1_xyz_images=cam1_xyz_images,
            config=extract_and_match_wrapper_config
        )
        time_tracker_init.add_time_stamp("ExtractAndMatchWrapper Initialisation")

    @staticmethod
    def get_creation_function(
            cam2_intrinsic_mtx:np.ndarray,
            extract_and_match_wrapper_config:ExtractAndMatchWrapperConfig = ExtractAndMatchWrapperConfig(),
    )->Callable[[RobotEnvironment,TimeTracker], 'OnlyPointsPredictor']:
        """
        Returns a function with which a new NoExtrasPredictor may be created.
        :param cam2_intrinsic_mtx: The 3x3 intrinsic matrix for camera 2
        :param extract_and_match_wrapper_config: The point cloud feature wrapper configuration
        :return: f(robot_env,time_tracker) -> NoExtrasPredictor
        """
        assert assert_intrinsic_mat(cam2_intrinsic_mtx)

        creation_function = lambda robot_env, init_tt: OnlyPointsPredictor(
            cam2_intrinsic_mtx=cam2_intrinsic_mtx.intrinsic_cam_mtx,
            cam1_bgr_images=robot_env.robot_bgr_images,
            cam1_xyz_images=robot_env.robot_xyz_images,
            extract_and_match_wrapper_config=extract_and_match_wrapper_config,
            time_tracker_init=init_tt
        )
        return creation_function

    
    def est_base_t_cam2(self,
                        cam2_bgr_image: np.ndarray,
                        number_retry:int = 1,
                        time_tracker:TimeTracker = TimeTracker()
                        ) -> np.ndarray | None:
        """
        Estimates the hom. transformation: baseT_cam2 based on only point features
        :param cam2_bgr_image: The camera 2 image (HxWx3-uint8 array)
        :param number_retry: With how many different cam1 images the prediction may be tried (upper bound)
        :param time_tracker: A time tracker where timestamps for the different subcomponents will be added.
        :return: The 4x4 Pose in SE3 if prediction was successful else None
        """
        assert assert_mxnx3_np_uint8_image(cam2_bgr_image)
        assert number_retry > 0

        time_tracker.reset_elapsed_time()
        base_t_cam =  self.extract_and_match_wrapper.est_base_t_cam2_with_retry(
            cam2_bgr_image=cam2_bgr_image,
            number_retry=number_retry,
        )
        time_tracker.add_time_stamp("extract and match wrapper call")
        return base_t_cam
    
    def update_pose(self,cam2_bgr_image: np.ndarray, rough_base_t_cam2:np.ndarray, time_tracker:TimeTracker) -> np.ndarray | None:
        """
        Acts exactly the same as est_base_t_cam2 with this predictor
        :param cam2_bgr_image: HxWx3 bgr image
        :param rough_base_t_cam2: A rough base_t_cam2 estimate.
        :param time_tracker: a time-tracker object, that will be used by the Pose Predictor to note the runtimes
        """
        return self.est_base_t_cam2(cam2_bgr_image=cam2_bgr_image, time_tracker=time_tracker, number_retry=1)



if __name__ == "__main__":
    robot_data = RobotEnvironment.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/pose_estimation/out_data_re")
    headset_data = HeadsetData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/pose_estimation/out_data_he")
    #robot_data = RobotEnvironment.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/out_data/rgbd_dataset_freiburg2_desk_robot_env")
    #headset_data = HeadsetData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/out_data/rgbd_dataset_freiburg2_desk_headset_data")
    
    
    
    predictor = OnlyPointsPredictor(
        cam2_intrinsic_mtx=headset_data.intrinsic_cam_mtx,
        cam1_bgr_images=robot_data.robot_bgr_images,
        cam1_xyz_images=robot_data.robot_xyz_images,
        extract_and_match_wrapper_config=ExtractAndMatchWrapperConfig(
            #extract_and_match=ExtractAndLightGlue(
            #    extractor="SuperPoint"
            #),
            extract_and_match=ExtractAndMatchLoMa('LoMaG'),
            rotation_augmentations= [Augmentation],
            ransac_config=pose_estimation_ransaac_config_less_precise,
        )
    )

    tt1 = TimeTracker()
    tt2 = TimeTracker()
    grader = OnePredictorRecordingGrader(
        predictor=predictor, 
        headset_data=headset_data,
        prediction_time_tracker=tt1,
        subcomponent_time_tracker=tt2
    )
    
    print(f"avg rot error: {np.round(np.rad2deg(grader.avg_rotational_error()), 2)} degrees")
    print(f"avg translational error: {np.round(grader.avg_translational_error()*1000, 1)} mm")
    print(f"median rot error: {np.round(np.rad2deg(grader.median_rotational_error()), 2)} degrees")
    print(f"median translational error: {np.round(grader.median_translational_error()*1000, 1)} mm")
    print(f"sucess_ratio: {np.round(grader.success_ratio(),2)}")

    grader.visualize_predictions(robot_env=robot_data, show_label=True)

    print(f"tt1:")
    tt1.print_report()
    print(f"\n tt2:")
    tt2.print_report()

    predictor.extract_and_match_wrapper.print_used_augmentations()
    print(f"avg number of tries: {predictor.extract_and_match_wrapper.avg_number_of_tries()}")