import sys, os

from image_camera_manipulation import create_3d_camera

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robot_environment import RobotEnvironment
from headset_data import HeadsetData
from assertion_helpers import *
from tqdm import tqdm
from time_tracker import TimeTracker
from abc import ABC, abstractmethod
import open3d as o3d
import matplotlib.pyplot as plt

class PosePredictor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def est_base_t_cam2(self,cam2_bgr_image: np.ndarray, number_retry:int = 2, time_tracker:TimeTracker = TimeTracker()) -> np.ndarray | None:
        """
        Predicts the homogenous transformation base_t_cam2.
        :param cam2_bgr_image: HxWx3-uint8 bgr image
        :param number_retry: The number of retries the predictor is allowed to do, until returning None
        :param time_tracker: a time-tracker object, that will be used by the Pose Predictor to note the runtimes
        :return: A 4x4 hom. transformation matrix: base T_cam2 or None if it fails.
        """
        return None
    
    @abstractmethod
    def update_pose(self,cam2_bgr_image: np.ndarray, rough_base_t_cam2:np.ndarray, time_tracker:TimeTracker) -> np.ndarray | None:
        """
        Some predictors can be faster/more efficient, when called via this function
        :param cam2_bgr_image: HxWx3 bgr image
        :param rough_base_t_cam2: A rough base_t_cam2 estimate.
        :param time_tracker: a time-tracker object, that will be used by the Pose Predictor to note the runtimes
        """
        return None

class OnePredictorRecordingGrader:
    def __init__(self,
                 predictor:PosePredictor,
                 headset_data:HeadsetData,
                 prediction_time_tracker:TimeTracker = TimeTracker(),
                 subcomponent_time_tracker:TimeTracker = TimeTracker()
                 ):
        """
        Creates an OnePredictorRecordingGrader, which is an object to assess the performance of a Predictor on a
        HeadsetData recording. It just uses simple `est_base_t_cam2` calls.
        :param predictor: An PosePredictor instance, that will be used on the headset_data
        :param headset_data: The headset data on which the predictor will be used
        :param prediction_time_tracker: A TimeTracker object that will be used by the OnePredictorRecordingGrader to stop the time per prediction
        :param subcomponent_time_tracker: A TimeTracker object that will be passed into the prediction calls
        :return: Nothing
        """
        self._predictor = predictor
        self._headset_data = headset_data

        prediction_time_tracker.reset_elapsed_time()
        predicted_b_t_h_s = []

        for  image in tqdm(headset_data.bgr_image_s):
            predicted_b_t_h_s.append(predictor.est_base_t_cam2(image, time_tracker=subcomponent_time_tracker))
            prediction_time_tracker.add_time_stamp("Single frame from scratch prediction")

        self.predicted_b_t_h_s = predicted_b_t_h_s

        self.predicted_b_t_h_s_not_none = [b_t_h for b_t_h in predicted_b_t_h_s if b_t_h is not None]


    def __str__(self):
        return f"{self._predictor} on {self._headset_data.name}"

    def translational_errors(self) -> list[float | None]:
        """
        :return: A list of Euclidean translational errors / None if not computable
        """
        translational_errors = []
        for predicted_b_t_h, b_t_h in zip(self.predicted_b_t_h_s, self._headset_data.robot_base_t_headset_s):
            if predicted_b_t_h is None or b_t_h is None:
                translational_errors.append(None)
            else:
                translational_errors.append(np.linalg.norm(predicted_b_t_h[:3,3] - b_t_h[:3,3]))
        return translational_errors

    def rotational_errors(self) -> list[float | None]:
        """
        :return: A list of rotational errors / None if not computable
        """
        rotational_errors = []
        for predicted_b_t_h, b_t_h in zip(self.predicted_b_t_h_s, self._headset_data.robot_base_t_headset_s):
            if predicted_b_t_h is None or b_t_h is None:
                rotational_errors.append(None)
            else:
                rotational_errors.append(rotational_difference(predicted_b_t_h, b_t_h))
        return rotational_errors

    def median_translational_error(self) -> float:
        """
        :return: The median of the Euclidean translational errors
        """
        if len(self.translational_errors()) == 0:
            return np.nan
        translational_errors_no_none = [e for e in self.translational_errors() if e is not None]
        return float(np.median(translational_errors_no_none))

    def median_rotational_error(self) -> float:
        """
        :return: The median of the rotational errors
        """
        if len(self.rotational_errors()) == 0:
            return np.nan
        rotational_errors_no_none = [e for e in self.rotational_errors() if e is not None]
        return float(np.median(rotational_errors_no_none))

    def avg_translational_error(self) -> float:
        """
        :return: The average of the Euclidean translational errors
        """
        if len(self.translational_errors()) == 0:
            return np.nan
        translational_errors_no_none = [e for e in self.translational_errors() if e is not None]
        return float(np.mean(translational_errors_no_none))

    def avg_rotational_error(self) -> float:
        """
        :return: The average of the rotational errors
        """
        if len(self.rotational_errors()) == 0:
            return np.nan
        rotational_errors_no_none = [e for e in self.rotational_errors() if e is not None]
        return float(np.mean(rotational_errors_no_none))

    def success_ratio(self)->float:
        """
        :return: The success ratio, so on how many frames a pose was predicted
        """
        return len(self.predicted_b_t_h_s_not_none)/len(self.predicted_b_t_h_s)

    def visualize_predictions(self, robot_env:RobotEnvironment|None = None, show_label:bool = False)->None:
        """
        Visualizes the predictions made by the predictor using open3d
        :param robot_env: RobotEnvironment or None, if not None will be added to the plot
        """
        colors = plt.cm.jet(np.linspace(0, 1, self._headset_data.n_frames))[:, :3]

        to_vis = []
        if robot_env is None:
            base_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.4)
            to_vis.append(base_frame)
        else:
            to_vis = robot_env.visualize_3d_data(visualize=False)

        for i, (predicted_b_t_h, b_t_h) in enumerate(zip(self.predicted_b_t_h_s, self._headset_data.robot_base_t_headset_s)):
            if b_t_h is not None and show_label:
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005)
                cam_frame.transform(b_t_h)
                to_vis.append(cam_frame)
                camera_line_set = create_3d_camera(
                    base_t_camera=b_t_h,
                    intrinsics=self._headset_data.intrinsic_cam_mtx,
                    hxw_img=self._headset_data.bgr_image_s[i],
                    scale=0.01
                )
                camera_line_set.paint_uniform_color(colors[i])
                to_vis.append(camera_line_set)

            if predicted_b_t_h is not None:
                cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.005)
                cam_frame.transform(predicted_b_t_h)
                to_vis.append(cam_frame)
                camera_line_set = create_3d_camera(
                    base_t_camera=predicted_b_t_h,
                    intrinsics=self._headset_data.intrinsic_cam_mtx,
                    hxw_img=self._headset_data.bgr_image_s[i],
                    scale=0.01
                )
                camera_line_set.paint_uniform_color(colors[i])
                to_vis.append(camera_line_set)

            if b_t_h is not None and predicted_b_t_h is not None and show_label:
                line_set = o3d.geometry.LineSet()
                line_set.points = o3d.utility.Vector3dVector([b_t_h[:3,3], predicted_b_t_h[:3,3]])
                line_set.lines = o3d.utility.Vector2iVector([[0,1]])
                line_set.paint_uniform_color(colors[i])
                to_vis.append(line_set)

        if len(self.predicted_b_t_h_s_not_none) > 0:
            pred_line_set = o3d.geometry.LineSet()
            pred_line_set.points = o3d.utility.Vector3dVector(np.array(self.predicted_b_t_h_s_not_none)[:,:3,3])
            pred_line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(self.predicted_b_t_h_s_not_none)-1)])
            pred_line_set.paint_uniform_color([1,0,0])
            to_vis.append(pred_line_set)

        obs_line_set = o3d.geometry.LineSet()
        points = np.array([m for m in self._headset_data.robot_base_t_headset_s if m is not None])[:, :3, 3]
        obs_line_set.points = o3d.utility.Vector3dVector(points)
        obs_line_set.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(points)-1)])
        obs_line_set.paint_uniform_color([0,1,0])
        to_vis.append(obs_line_set)

        o3d.visualization.draw_geometries(to_vis, f"Headset Predictions visualization")
