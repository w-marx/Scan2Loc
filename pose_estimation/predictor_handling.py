import numpy as np

from preprocessing_2_prediction import PredictionData
from no_extras_predictor import NoExtrasPredictor


class PosePredictor:
    def __init__(self):
        pass

    def predict_poses(self,
                      cam1_bgr_image:np.ndarray,
                      cam1_xyz_image:np.ndarray,
                      cam2_bgr_image: np.ndarray,
                      point_cloud:np.ndarray,
                      ) -> np.ndarray | None:
        """
        Predicts the homogenous transformation cam1_t_cam2
        """
        raise NotImplementedError


def grade_predictions(predictions: list[np.ndarray], actual: list[np.ndarray]) -> tuple[list[float|None], list[float|None]]:

    rotational_errors = []
    translational_errors = []

    for prediction, actual in zip(predictions, actual):
        if prediction is None:
            rotational_errors.append(None)
            translational_errors.append(None)

        rotational_dist = np.arccos((np.trace(prediction, np.linalg.inv(actual))-1)/2)
        euclidean_dist = np.linalg.norm(prediction - actual)

        rotational_errors.append(rotational_dist)
        translational_errors.append(euclidean_dist)

    return rotational_errors, translational_errors



def run_predictions(data:PredictionData, predictor:PosePredictor) -> list[np.ndarray | None]:

    point_cloud = data.point_cloud
    headset_image = data.headset_bgr_image
    robot_bgr_images = data.robot_bgr_images
    robot_xyz_images = data.robot_xyz_images
    robot_base_t_robot_cam_s = data.robot_base_t_robot_camera_s


    predicted_poses = []

    for xyz_img, bgr_img, b_t_c in zip(robot_xyz_images, robot_bgr_images, robot_base_t_robot_cam_s):
        cam_robot_cam_t_headset_cam = predictor.predict_poses(
            cam2_bgr_image=headset_image,
            cam1_bgr_image=bgr_img,
            cam1_xyz_image=xyz_img,
            point_cloud=point_cloud
        )
        if cam_robot_cam_t_headset_cam is None:
            predicted_poses.append(None)
        predicted_poses.append(b_t_c @ cam_robot_cam_t_headset_cam)

    return predicted_poses



if __name__ == "__main__":
    data = PredictionData.from_folder("../preprocessed_data/r2_aruco2")
    predictor = NoExtrasPredictor(data.headset_intrinsics)
    predicted_poses = run_predictions(data, predictor)

    if data.robot_base_t_headset is not None:
        r_err, t_err = grade_predictions(predicted_poses, data.robot_base_t_headset)
        print(f"r_err: {r_err} \n\n t_err: {t_err}")