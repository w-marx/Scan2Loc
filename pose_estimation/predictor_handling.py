import numpy as np

class PosePredictor:
    def __init__(self):
        pass

    def predict_poses(self,
                      headset_image:np.ndarray,
                      robot_rgb_image:np.ndarray,
                      robot_xyz_image:np.ndarray,
                      robot_base_t_robot_cam:np.ndarray,
                      point_cloud:np.ndarray,
                      ) -> np.ndarray:
        pass


def grade_predictions(predictions: list[np.ndarray], actual: list[np.ndarray]) -> tuple[list[float], list[float]]:

    rotational_errors = []
    translational_errors = []

    for prediction, actual in zip(predictions, actual):

        rotational_dist = np.arccos((np.trace(prediction, np.linalg.inv(actual))-1)/2)
        euclidean_dist = np.linalg.norm(prediction - actual)

        rotational_errors.append(rotational_dist)
        translational_errors.append(euclidean_dist)

    return rotational_errors, translational_errors



def run_predictions(data:PredictionData, predictor:PosePredictor) -> list[np.ndarray]:

    point_cloud: np.ndarray = input_dict["point_cloud"]
    headset_image: np.ndarray = input_dict["headset_image"]

    predicted_poses = []

    for datapoint in input_dict["robot_datapoints"]:
        predicted_poses.append(predictor.predict_poses(
            headset_image=headset_image,
            robot_rgb_image=datapoint["rgb_image"],
            robot_xyz_image=datapoint["xyz"],
            robot_base_t_robot_cam=datapoint["robot_base_t_robot_cam"],
            point_cloud=point_cloud
        ))

    return predicted_poses



