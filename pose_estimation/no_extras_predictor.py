import cv2
import numpy as np
from sklearn.externals.array_api_compat import torch

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from pose_estimation.predictor_handling import *



class NoExtrasPredictor(PosePredictor):
    def __init__(self, cam2_mtx:np.ndarray):
        super().__init__()

        self.cam2_mtx = cam2_mtx
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        self.matcher = LightGlue(features='superpoint').eval().cuda()

    def predict(self,
            cam1_bgr_image:np.ndarray,
            cam1_xyz_image:np.ndarray,
            cam2_bgr_image: np.ndarray,
            point_cloud:np.ndarray,
        ) -> np.ndarray | None:

        cam2_image = torch.tensor(cam2_bgr_image)
        robot_image = torch.tensor(cam1_bgr_image)

        feats_headset = self.extractor.extract(cam2_image)  # auto-resize the image, disable with resize=None
        feats_robot = self.extractor.extract(robot_image)

        matches01 = self.matcher({'cam2_image': feats_headset, 'robot_image': feats_robot})
        feats_headset, feats_robot, matches01 = [rbd(x) for x in [feats_headset, feats_robot, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)

        image_points_headset = feats_headset['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        image_points_robot = feats_robot['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
        object_points_robot = [cam1_xyz_image[x,y] for x,y in image_points_robot]

        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points_robot, image_points_headset, self.cam2_mtx, [0,0,0,0,0], flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
           return None

        transformation = np.eye(4)
        transformation[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        transformation[:3, 3] = translation_vector.flatten()
        
        return transformation

