import cv2
import numpy as np
from predictor_handling import *
from extractors_and_matchers import *
from shared.assertion_helpers import assert_intrinsic_mat

from shared.assertion_helpers import assert_mxn_np_float_image_batch, assert_mxnx3_np_uint8_image_batch

from robot_environment import RobotEnvironment

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
            cam2_intrinsic_mtx=cam2_intrinsic_mtx,
            cam1_bgr_images=robot_env.robot_bgr_images,
            cam1_xyz_images=robot_env.robot_xyz_images,
            extract_and_match_wrapper_config=extract_and_match_wrapper_config,
            time_tracker_init=init_tt
        )
        return creation_function

    
    def est_base_t_cam2(self,
                        cam2_bgr_image: np.ndarray,
                        number_retry:int = 1,
                        time_tracker:TimeTracker = TimeTracker(),
                        fd:FeatureDrawing | None = None
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
            fd = fd
        )
        time_tracker.add_time_stamp("extract and match wrapper call")
        return base_t_cam