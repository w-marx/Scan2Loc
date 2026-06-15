from typing import Literal
from PIL import Image
import numpy as np
import cv2, sys, os, torch
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from shared.assertion_helpers import *

from .geometric_utilities.ransac_pose_estimation import * 
from .small_utilities.image_augmentation import * 
from .small_utilities.sheduler import *

class ExtractAndMatch(ABC):

    @abstractmethod
    def get_features(self, img_rgb:np.ndarray):
        """
        Will return the features for this image
        """
        pass

    @abstractmethod
    def match_features(self, features1, features2) -> tuple[np.ndarray, np.ndarray]:
        """
        Will match the features and return image coor
        :param features1: Features extracted from one image using `get_features`
        :param features2: Features extracted from another image using `get_features`
        :return: a tuple of image Points as 2 Nx2 numpy arrays (in the x-y format)
        """
        pass

    def get_matched_points(self,img1_rgb:np.ndarray, img2_rgb:np.ndarray, plot_results:bool = False)->tuple[np.ndarray, np.ndarray]:
        """
        :param img1_rgb: An RGB image as HxWx3-uint8 numpy array
        :param img2_rgb: An RGB image as HxWx3-uint8 numpy array
        :param plot_results: wheather to plot the matched features
        :return: a tuple of image Points as 2 Nx2 numpy arrays (in the x-y format)
        """
        points1, points2 = self.match_features(self.get_features(img1_rgb), self.get_features(img2_rgb))
        if plot_results:
            self.plot_matched_points(
                img1_rgb=img1_rgb, img2_rgb=img2_rgb, points1=points1, points2=points2
            )
        return points1, points2


    @staticmethod
    def plot_matched_points(img1_rgb:np.ndarray, img2_rgb:np.ndarray, points1:np.ndarray, points2:np.ndarray):
        """
        :param img1_rgb: An RGB image as HxWx3-uint8 numpy array
        :param img2_rgb: An RGB image as HxWx3-uint8 numpy array
        :param points1: Nx2 array of 2d points of the form [[x1,y1], ...] in img1_rgb points1[i] is matched to points2[i]
        :param points1: Nx2 array of 2d points
        :return: a tuple of image Points as 2 Nx2 numpy arrays (in the x-y format)
        """
        _ = assert_mxnx3_np_uint8_image(img1_rgb)
        _ = assert_mxnx3_np_uint8_image(img2_rgb)
        assert points1.ndim == 2 and points1.shape[-1] == 2, f"invalid 2d points shape: {points1.shape}"
        assert points2.ndim == 2 and points2.shape[-1] == 2, f"invalid 2d points shape: {points2.shape}"
        assert points1.shape == points2.shape, f"Incompatible shapes for matched: {points1.shape} != {points2.shape}"

        h1,w1 = img1_rgb.shape[:2]
        h2,w2 = img2_rgb.shape[:2]
        x_offset = 10
        cnvs_h, cnvs_w = max(h1, h2), w1+w2+x_offset
        h1_off, h2_off = int((cnvs_h-h1)/2), int((cnvs_h-h2)/2)

        canvas = np.zeros((cnvs_h, cnvs_w, 3), dtype = np.uint8)
        canvas[h1_off:h1+h1_off, :w1] = img1_rgb
        canvas[h2_off:h2+h2_off, w1+x_offset:cnvs_w] = img2_rgb

        fig, ax = plt.subplots(figsize = (12, 8))
        ax.set_title("Matched image points")
        ax.imshow(canvas)

        colors = plt.cm.jet(np.linspace(0,1, points1.shape[0]))

        for i, ((x1,y1), (x2, y2)) in enumerate(zip(points1, points2)):

            ax.scatter(x1, h1_off+y1, color=colors[i], s=5, alpha=0.8)
            ax.scatter(x2+w1+x_offset, h2_off+y2, color=colors[i], s=5, alpha=0.8)

            ax.plot(
                [x1, x2+w1+x_offset,],
                [y1+h1_off, y2+h2_off],
                color=colors[i],
                linewidth=1,
                alpha = 0.5
            )
        ax.axis('off')
        plt.show()

class ExtractAndLightGlue(ExtractAndMatch):
    def __init__(
        self, 
        extractor:Literal["SuperPoint", "DISK", "SIFT", "ALIKED", "DogHardNet"] = "SuperPoint",
        max_num_keypoints:int = 2048
    ):
        """
        A LightGlue based ExtractAndMatch Class
        :param extractor: What extractor to use before Lightglue
        :param max_num_keypoints: the maximum number of keypoints for the extractors
        """

        from lightglue import LightGlue, SuperPoint, DISK, SIFT, ALIKED, DoGHardNet
        from lightglue.utils import rbd, numpy_image_to_torch

        extractors = {
            "SuperPoint": SuperPoint,
            "DISK":DISK,
            "SIFT": SIFT,
            "ALIKED": ALIKED,
            "DogHardNet": DoGHardNet
        }# TODO add ORB (looks like not possible)

        assert extractor in extractors, f"Extractor: {extractor} is not supported"
        assert 0 < max_num_keypoints, f"max_num_keypoints negative: {max_num_keypoints}"

        self.feature_name = extractor.lower()

        self._rbd = rbd
        self._numpy_image_to_torch = numpy_image_to_torch

        self.extractor = extractors[extractor](max_num_keypoints=max_num_keypoints).eval().cuda()
        self.matcher = LightGlue(features=self.feature_name).eval().cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_features(self, img_rgb:np.ndarray):
        assert assert_mxnx3_np_uint8_image(img_rgb)

        torch_image = self._numpy_image_to_torch(img_rgb)
        feats_cam1 = self.extractor.extract(torch_image.to(self.device))
        return feats_cam1
    
    def match_features(self, features1, features2) -> tuple[np.ndarray, np.ndarray]:
        matches12 = self.matcher({'image0': features1, 'image1': features2, })
        feats_cam1, feats_cam2, matches12 = [self._rbd(x) for x in [features1, features2, matches12]]

        feats_cam1_keypoints = feats_cam1['keypoints']
        feats_cam2_keypoints = feats_cam2['keypoints']
        image_points_cam1_cpu_np = feats_cam1_keypoints[matches12['matches'][..., 0]].cpu().numpy()
        image_points_cam2_cpu_np = feats_cam2_keypoints[matches12['matches'][..., 1]].cpu().numpy()

        return image_points_cam1_cpu_np, image_points_cam2_cpu_np
    

class ExtractAndMatchLoMa(ExtractAndMatch):
    def __init__(self, loma_variant:Literal["LoMaB", "LoMaB128", "LoMaL", "LoMaG", "LoMaR"] = "LoMaG"):
        """
        LoMa based ExtractAndMatch class
        :param loma_variant:
        """
        from loma import LoMa, LoMaB, LoMaB128, LoMaL, LoMaG, LoMaR
        loma_variant_s = {
            "LoMaB": LoMaB,
            "LoMaB128":LoMaB128,
            "LoMaL": LoMaL,
            "LoMaG": LoMaG,
            "LoMaR": LoMaR
        }
        assert loma_variant in loma_variant_s, f"The Loma variant:{loma_variant} is not supported"

        self.model = LoMa(loma_variant_s[loma_variant])

    def get_features(self, img_rgb:np.ndarray):
        """
        :param img_rgb: An HxWx3-unint8 RGB image
        :return: an torch tensor of the image with height & width %14 = 0
        """
        assert assert_mxnx3_np_uint8_image(img_rgb)

        h1, w1 = (img_rgb.shape[0] // 14)*14, (img_rgb.shape[1] // 14)*14
        img_rgb_m14 = img_rgb[:h1, : w1, :]
        img_tensor = torch.from_numpy(img_rgb_m14).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        return img_tensor
    

    def match_features(self, features1, features2) -> tuple[np.ndarray, np.ndarray]:
        """
        :param features1: A torch tensor of a image with height & width %14 = 0
        :param features2: Another torch tensor of a image with height & width %14 = 0
        :return: a tuple of image Points as 2 Nx2 numpy arrays (in the x-y format)
        """
        kpts1, kpts2 = self.model.match(features1, features2)
        return kpts1, kpts2

class ExtractAndMatchEffLoFTR(ExtractAndMatch):
    def __init__(self, matching_threshhold:float = 0.3):
        """
        Extract and match based on efficient LoFTR
        :param matching_threshhold: filter threshhold int [0,1] for matchings
        """
        assert 0 <= matching_threshhold <= 1, f"invalid threshhold: {matching_threshhold} not in [0,1]"

        import transformers
        from transformers import AutoImageProcessor, AutoModelForKeypointMatching
        self.processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr") 
        self.model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")
        self.matching_threshhold = matching_threshhold

    def get_features(self, img_rgb:np.ndarray):
        """
        Turns the rgb image into an pil image
        """
        assert assert_mxnx3_np_uint8_image(img_rgb)
        return Image.fromarray(img_rgb)
    

    def match_features(self, features1, features2) -> tuple[np.ndarray, np.ndarray]:
        """
        :param features1: A PIL Image
        :param features2: Another PIL Image
        :return: a tuple of image Points as 2 Nx2 numpy arrays (in the x-y format)
        """
        inputs = self.processor([features1, features2], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        image_sizes = [[(features1.height, features1.width),(features2.height, features2.width)]]
        output = self.processor.post_process_keypoint_matching(outputs, image_sizes, threshold=self.matching_threshhold)[0]

        kpts1 = output["keypoints0"].cpu().numpy().astype(np.float32)
        kpts2 = output["keypoints1"].cpu().numpy().astype(np.float32)
        return kpts1, kpts2

@dataclass
class ExtractAndMatchWrapperConfig:
    extract_and_match:ExtractAndMatch = field(
        default_factory=ExtractAndLightGlue
    )
    rotation_augmentations:list[type[Augmentation]] = field(
        default_factory=lambda: [Rotate180Deg]
    )
    crop_augmentations:list[float] | None = None
    ransac_config:RansacPoseEstimationConfig = pose_estimation_ransaac_config_precise
    sheduler:type[Sheduler] = EMASheduler
    display_matching:bool = False



class ExtractAndMatchWrapper:
    def __init__(
            self,
            cam2_mtx:np.ndarray,
            cam1_bgr_images:np.ndarray,
            cam1_xyz_images:np.ndarray,
            config:ExtractAndMatchWrapperConfig
        ):

        self.cam2_mtx = cam2_mtx
        self.extract_and_match = config.extract_and_match
        self.ransac_config = config.ransac_config

        self.rotation_augmentations = [aug_class() for aug_class in config.rotation_augmentations]
        
        self.crop_augmentations = [Augmentation()]
        if config.crop_augmentations is not None:
            self.crop_augmentations += [CropImage(x) for x in config.crop_augmentations if 0 <= x < 1.0]  

        self.cam1_features = [self.extract_and_match.get_features(img) for img in cam1_bgr_images]
        self.cam1_xyz_images = cam1_xyz_images

        self.sheduler = config.sheduler(cam1_bgr_images.shape[0])

        # For debugging/additional information
        self.chosen_augmentations = defaultdict(int)
        self.used_number_of_tries = []

        self.display_matching = config.display_matching
        self.cam1_bgr_images_for_vis = cam1_bgr_images if config.display_matching else None

    def get_sheduler(self):
        return self.sheduler
    

    def est_base_t_cam2_with_retry(self,cam2_bgr_image: np.ndarray, number_retry:int = 1, fd:FeatureDrawing|None = None) -> np.ndarray | None:
        """
        Will try to match points until a pose is found or number_retry was reached
        """
        base_t_cam_w_points = self.est_base_t_cam2_and_points_with_retry(
            cam2_bgr_image=cam2_bgr_image, 
            number_retry=number_retry,
            fd = fd
        )
        if base_t_cam_w_points is None:
            return None
        return base_t_cam_w_points[0]
    

    def est_base_t_cam2_and_points_with_retry(
            self,
            cam2_bgr_image: np.ndarray, 
            number_retry:int = 1,
            fd:FeatureDrawing | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Will try to match points until a pose is found or number_retry was reached
        :return None or base T_cam, points_image_1, points_image_2, world_obj_points, inliers
        """
        number_tries = 0
        est_base_t_cam_and_points = None
        cam2_rgb_image = cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)

        while est_base_t_cam_and_points is None and number_tries < number_retry:
            idx = self.sheduler.get_best()
            est_base_t_cam_and_points = self.est_base_t_cam2_and_points(
                idx=idx,
                cam2_rgb_image = cam2_rgb_image,
                fd=fd
            )
            self.sheduler.adjust(idx, est_base_t_cam_and_points is not None)
            number_tries += 1
        self.used_number_of_tries.append(number_tries)
        return est_base_t_cam_and_points


    def est_base_t_cam2_and_points(
            self,
            idx:int,
            cam2_rgb_image: np.ndarray,
            fd:FeatureDrawing | None = None
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]] | None:
        """
        :return: None or base T_cam, points_image_1, points_image_2, world_obj_points, inliers
        """

        best_num_of_points = 0
        best_aug_option_name = ""
        best_image_points_cam1 = None
        best_image_points_cam2 = None
        best_world_obj_points = None
        augmented_image_point_4_vis = None

        for c_aug in self.crop_augmentations:
            for r_aug in self.rotation_augmentations:
                augmented_image, backward_aug2 = c_aug.forward(cam2_rgb_image)
                augmented_image, backward_aug1 = r_aug.forward(augmented_image)

                features_cam2 = self.extract_and_match.get_features(augmented_image)
                image_points_cam1, image_points_cam2 = self.extract_and_match.match_features(
                    features1=self.cam1_features[idx],
                    features2=features_cam2
                )

                if image_points_cam1.shape[0] < max(self.ransac_config.min_number_inlier_afterwards,6):
                    continue

                if image_points_cam1.shape[0] < best_num_of_points:
                     continue


                world_obj_points = np.array([self.cam1_xyz_images[idx][int(np.round(y)),int(np.round(x))] for x,y in image_points_cam1])
                valid_points_mask = np.isfinite(world_obj_points).all(axis=1)

                if np.sum(valid_points_mask) > best_num_of_points:
                    best_num_of_points = np.sum(valid_points_mask)
                    best_aug_option_name = f"{c_aug} x {r_aug}"
                    best_image_points_cam1 = image_points_cam1[valid_points_mask]
                    best_image_points_cam2 = backward_aug2(backward_aug1(image_points_cam2))[valid_points_mask]
                    best_world_obj_points = world_obj_points[valid_points_mask]

                    if self.display_matching:
                        augmented_image_point_4_vis = (augmented_image, image_points_cam2[valid_points_mask])


        if best_num_of_points < max(self.ransac_config.min_number_inlier_afterwards,6):
            return None

        self.chosen_augmentations[best_aug_option_name] += 1

        cam2_t_base__inliers = estimate_point_pose_ransac(
            world_points=best_world_obj_points,
            img_points=best_image_points_cam2,
            intrinsic_matrix=self.cam2_mtx,
            config=self.ransac_config,
            fd = fd
        )

        if cam2_t_base__inliers is None:
            return None
        
        if self.display_matching:
            aug_img, points2 = augmented_image_point_4_vis
            ExtractAndMatch.plot_matched_points(
                img1_rgb=cv2.cvtColor(self.cam1_bgr_images_for_vis[idx], code=cv2.COLOR_BGR2RGB),
                points1=best_image_points_cam1,
                img2_rgb=aug_img,
                points2=points2
            )

        return np.linalg.inv(cam2_t_base__inliers[0]), best_image_points_cam1, best_image_points_cam2, best_world_obj_points, cam2_t_base__inliers[1]


    def est_base_t_cam2(self,
                        idx:int,
                        cam2_rgb_image: np.ndarray,
                        ) -> np.ndarray | None:
        base_t_cam_w_points = self.est_base_t_cam2_and_points(
            idx=idx,
            cam2_rgb_image=cam2_rgb_image
        )
        return None if base_t_cam_w_points is None else base_t_cam_w_points[0]
    
    def print_used_augmentations(self):
        print(f"Chosen augmentations:")
        for name, count in sorted(self.chosen_augmentations.items(), key=lambda x: -x[1]):
            print(f" {name:<40}  {count}")
    
    def avg_number_of_tries(self)->float:
        return np.mean(self.used_number_of_tries)