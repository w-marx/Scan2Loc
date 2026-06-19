from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from shared.se3_utilities import *
from shared.assertion_helpers import *


from .geometric_utilities.slam2mp4 import FeatureDrawing
from .geometric_utilities.time_tracker import TimeTracker
from .extractors_and_matchers import ExtractAndMatchWrapper


class PosePredictor(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def est_base_t_cam2(self,cam2_bgr_image: np.ndarray, number_retry:int = 1, time_tracker:TimeTracker = TimeTracker(), fd:FeatureDrawing | None = None) -> np.ndarray | None:
        """
        Predicts the homogenous transformation base_t_cam2.
        :param cam2_bgr_image: HxWx3-uint8 bgr image
        :param number_retry: The number of retries the predictor is allowed to do, until returning None
        :param time_tracker: a time-tracker object, that will be used by the Pose Predictor to note the runtimes
        :param axes: An matplotlib axes object on which the used features will be drawn if not None (bad for performance)
        :return: A 4x4 hom. transformation matrix: base T_cam2 or None if it fails.
        """
        return None
    
    @property
    def extract_and_match_wrapper(self)->ExtractAndMatchWrapper | None:
        return None