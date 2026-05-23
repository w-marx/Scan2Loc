import torch
from typing import Literal, Callable
import numpy as np
import cv2

class ExtractAndMatch:
    def get_matched_points(self,img1_rgb:np.ndarray, img2_rgb:np.ndarray)->tuple[np.ndarray, np.ndarray]:
        """
        :param img1_rgb: An RGB image as HxWx3-uint8 numpy array
        :param img2_rgb: An RGB image as HxWx3-uint8 numpy array
        :return: a tuple of image Points as 2 Nx2 numpy arrays (in the x-y format)
        """
        raise NotImplementedError("Not implemented in base class")
    
    def plot_matched_points(img1:np.ndarray, img2:np.ndarray, points1:np.ndarray, points2:np.ndarray):
        raise NotImplementedError("Not implemented yet")

class ExtractAndLightGlue(ExtractAndMatch):
    def __init__(
        self, 
        extractor:Literal["SuperPoint", "DISK", "SIFT", "ALIKED", "DogHardNet"] = "SuperPoint",
        max_num_keypoints:int = 2048
    ):
        super().__init__()
        from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
        from lightglue.utils import rbd, numpy_image_to_torch

        extractors = {
            "SuperPoint": SuperPoint,
            "DISK":DISK,
            "SIFT": SIFT,
            "ALIKED": ALIKED,
            "DogHardNet": DoGHardNet
        }# TODO add ORB

        self.feature_name = extractor.lower()

        self._rbd = rbd
        self._numpy_image_to_torch = numpy_image_to_torch

        self.extractor = extractors[extractor](max_num_keypoints=max_num_keypoints).eval().cuda()
        self.matcher = LightGlue(features=self.feature_name).eval().cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_matched_points(self,img1_rgb:np.ndarray, img2_rgb:np.ndarray)->tuple[np.ndarray, np.ndarray]:
            cam1_image = self._numpy_image_to_torch(img1_rgb)
            cam2_image = self._numpy_image_to_torch(img2_rgb)

            feats_cam1 = self.extractor.extract(cam1_image.to(self.device))
            feats_cam2 = self.extractor.extract(cam2_image.to(self.device))

            matches12 = self.matcher({'image0': feats_cam1, 'image1': feats_cam2, })
            feats_cam1, feats_cam2, matches12 = [self._rbd(x) for x in [feats_cam1, feats_cam2, matches12]]

            feats_cam1_keypoints = feats_cam1['keypoints']
            feats_cam2_keypoints = feats_cam2['keypoints']
            image_points_cam1_cpu_np = feats_cam1_keypoints[matches12['matches'][..., 0]].cpu().numpy()
            image_points_cam2_cpu_np = feats_cam2_keypoints[matches12['matches'][..., 1]].cpu().numpy()

            return image_points_cam1_cpu_np, image_points_cam2_cpu_np

class ExtractAndMatchLoMa(ExtractAndMatch):
    def __init__(self, loma_variant:Literal["LoMaB", "LoMaB128", "LoMaL", "LoMaG", "LoMaR"] = "LoMaG"):
        from loma import LoMa, LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR
        extractors = {
            "LoMaB": LoMaB,
            "LoMaB128":LoMaB128,
            "LoMaL": LoMaL,
            "LoMaG": LoMaG,
            "LoMaR": LoMaR
        }
        self.model = LoMa(extractors[loma_variant])

    def get_matched_points(self,img1_rgb:np.ndarray, img2_rgb:np.ndarray)->tuple[np.ndarray, np.ndarray]:
        h1, w1 = (img1_rgb.shape[0] // 14)*14, (img1_rgb.shape[1] // 14)*14
        img1_rgb_m14 = img1_rgb[:h1, : w1, :]

        h2, w2 = (img2_rgb.shape[0] // 14)*14, (img2_rgb.shape[1] // 14)*14
        img2_rgb_m14 = img2_rgb[:h2, : w2, :]
    
        img1_tensor = torch.from_numpy(img1_rgb_m14).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2_tensor = torch.from_numpy(img2_rgb_m14).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        kpts1, kpts2 = self.model.match(img1_tensor, img2_tensor)
        return kpts1, kpts2


class Augmentation:
    def __str__(self) -> str: return "Identity"
    def forward(self,img:np.ndarray)->tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        return img, lambda x:x

class Rotate180Deg(Augmentation):
    def __str__(self)-> str: return "Rotate 180°"
    def forward(self,img:np.ndarray)->tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        w, h = img.shape[:2]
        img = cv2.rotate(img, cv2.ROTATE_180)
        backward = lambda img_points: np.array([[w-x, h-y] for x,y in img_points])
        return img, backward

class CropImage(Augmentation):
    def __init__(self, relative_crop_amount:float = 0):
        self.relative_crop_amount = relative_crop_amount
    
    def __str__(self)->str: return f"CropImage {self.relative_crop_amount}"

    def forward(self,img:np.ndarray)->tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        crop_pixels = int(self.relative_crop_amount * np.min(img.shape[:2])/2)
        img = img[crop_pixels:-crop_pixels,crop_pixels:-crop_pixels,:]
        backward = lambda img_points: img_points + crop_pixels
        return img, backward