import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocessing_2_prediction import PredictionData, create_3d_camera


class PosePredictor:
    def __init__(self):
        pass

    def est_base_t_cam2(self,
                        cam1_bgr_image:np.ndarray,
                        base_xyz_image:np.ndarray,
                        cam2_bgr_image: np.ndarray,
                        point_cloud:np.ndarray,
                        ) -> np.ndarray | None:
        """
        Predicts the homogenous transformation base_t_cam2
        """
        raise NotImplementedError


def grade_predictions(predictions: list[np.ndarray], actual: np.ndarray) -> tuple[list[float|None], list[float|None]]:

    calc_rotational_difference = lambda x, y: np.arccos((np.trace(x[:3, :3] @ y[:3, :3].T) - 1) / 2)

    rotational_errors = []
    translational_errors = []

    for prediction in predictions:
        if prediction is None:
            rotational_errors.append(None)
            translational_errors.append(None)
            continue

        trans_dist = np.linalg.norm(prediction[:3,3]-actual[:3,3])
        rotational_dist = calc_rotational_difference(prediction[:3,:3], actual[:3,:3])

        translational_errors.append(trans_dist)
        rotational_errors.append(rotational_dist)

    return rotational_errors, translational_errors



def run_predictions(data:PredictionData, predictor:PosePredictor) -> list[np.ndarray | None]:

    point_cloud = data.point_cloud
    headset_image = data.headset_bgr_image
    robot_bgr_images = data.robot_bgr_images
    robot_xyz_images = data.robot_xyz_images
    robot_base_t_robot_cam_s = data.robot_base_t_robot_camera_s

    predicted_poses = []

    for xyz_img, bgr_img, rb_t_rc in zip(robot_xyz_images, robot_bgr_images, robot_base_t_robot_cam_s):
        base_t_headset_cam = predictor.est_base_t_cam2(
            cam2_bgr_image=headset_image,
            cam1_bgr_image=bgr_img,
            base_xyz_image=xyz_img,
            point_cloud=point_cloud
        )
        if base_t_headset_cam is None:
            predicted_poses.append(None)
            continue
        predicted_poses.append(base_t_headset_cam)

    return predicted_poses