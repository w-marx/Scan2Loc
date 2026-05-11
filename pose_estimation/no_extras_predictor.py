import cv2
import numpy as np
from sklearn.externals.array_api_compat import torch

from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
from lightglue.utils import load_image, rbd
from pose_estimation.predictor_handling import *



class NoExtrasPredictor(PosePredictor):
    def __init__(self, headset_cam_mtx:np.ndarray, headset_cam_dist:np.ndarray):
        super().__init__()

        self.headset_cam_mtx = headset_cam_mtx
        self.headset_cam_dist = headset_cam_dist

        self.extractor = SuperPoint(max_num_keypoints=2048).eval().cuda()
        self.matcher = LightGlue(features='superpoint').eval().cuda()

    def predict(self, rgb_image_headset: np.ndarray, rgb_image_robot:np.ndarray, xyz_image_robot:np.ndarray) -> np.ndarray:
        headset_image = torch.tensor(rgb_image_headset)
        robot_image = torch.tensor(rgb_image_robot)

        feats_headset = self.extractor.extract(headset_image)  # auto-resize the image, disable with resize=None
        feats_robot = self.extractor.extract(robot_image)

        matches01 = self.matcher({'headset_image': feats_headset, 'robot_image': feats_robot})
        feats_headset, feats_robot, matches01 = [rbd(x) for x in [feats_headset, feats_robot, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)

        image_points_headset = feats_headset['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        image_points_robot = feats_robot['keypoints'][matches[..., 1]]  # coordinates in image #1, shape (K,2)
        object_points_robot = [xyz_image_robot[x,y] for x,y in image_points_robot]

        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points_robot, image_points_headset, self.headset_cam_mtx, self.headset_cam_dist, flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            Exception("Could not solve PNP")

        transformation = np.eye(4)
        transformation[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
        transformation[:3, 3] = translation_vector.flatten()
        
        return transformation

