from typing import SupportsFloat, Any, Callable
import numpy as np
import cv2
from dataclasses import dataclass, field
from collections import defaultdict

from .extractors_and_matchers import ExtractAndMatch, ExtractAndLightGlue
from .ransac_pose_estimation import RansacPoseEstimationConfig, pose_estimation_ransaac_config_precise, estimate_point_pose_ransac
from .image_augmentation import Augmentation, Rotate180Deg, CropImage
from .sheduler import Scheduler, BlockingEMAScheduler

from ..utilities.slam2mp4 import FeatureDrawing


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
    scheduler:type[Scheduler] = BlockingEMAScheduler
    display_matching:bool = False

    def __post_init__(self):
        assert self.crop_augmentations is None or all([0 <= c < 1.0 for c in self.crop_augmentations]), f"Invalid: {self.crop_augmentations}"



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
            self.crop_augmentations = [CropImage(x) for x in config.crop_augmentations]  

        self.cam1_features = [self.extract_and_match.get_features(img) for img in cam1_bgr_images]
        self.cam1_xyz_images = cam1_xyz_images

        self.sheduler = config.scheduler(cam1_bgr_images.shape[0])

        # For debugging/additional information
        self.debug_chosen_augmentations = defaultdict(int)
        self.debug_used_number_of_tries = []
        self.debug_number_inliers = []

        self.display_matching = config.display_matching
        self.cam1_bgr_images_for_vis = cam1_bgr_images


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
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]] | None:
        """
        Will try to match points until a pose is found or number_retry was reached
        :return None or base T_cam, points_image_1, points_image_2, world_obj_points, inliers
        """
        number_tries = 0
        est_base_t_cam_and_points = None
        cam2_rgb_image = cv2.cvtColor(cam2_bgr_image, cv2.COLOR_BGR2RGB)

        augmented_images = []
        augmented_images_features = []
        map_points_to_unaugmented_functions_and_names = []

        for c_aug in self.crop_augmentations:
            for r_aug in self.rotation_augmentations:
                augmented_image, backward_aug2 = c_aug.forward(cam2_rgb_image)
                augmented_image, backward_aug1 = r_aug.forward(augmented_image)

                augmented_images.append(augmented_image)
                augmented_images_features.append(self.extract_and_match.get_features(augmented_image))
                map_points_to_unaugmented_functions_and_names.append(
                    (f"{c_aug} x {r_aug}", lambda points: backward_aug2(backward_aug1(points)))
                )

        while est_base_t_cam_and_points is None and number_tries < number_retry:
            idx = self.sheduler.get_best()
            est_base_t_cam_and_points = self.est_base_t_cam2_and_points(
                idx=idx,
                cam2_rgb_image_features = augmented_images_features,
                backward_transformations_names=map_points_to_unaugmented_functions_and_names,
                augmented_images=np.asarray(augmented_images),
            )

            if fd is not None:

                if est_base_t_cam_and_points is not None:
                    b_t_c, best_image_points_cam1, best_image_points_cam2, points1_3d, inlier_indices = est_base_t_cam_and_points
                    fd.visualize_localizer(
                        robot_img_rgb=cv2.cvtColor(self.cam1_bgr_images_for_vis[idx], code=cv2.COLOR_BGR2RGB),
                        headset_img_rgb=cam2_rgb_image,
                        points1=best_image_points_cam1[inlier_indices],
                        points2=best_image_points_cam2[inlier_indices],
                        points1_3d=points1_3d[inlier_indices],
                        base_t_cam=b_t_c,
                        headset_intrinsic_mat=self.cam2_mtx
                    )
                else:
                    fd.visualize_localizer(
                        robot_img_rgb=cv2.cvtColor(self.cam1_bgr_images_for_vis[idx], code=cv2.COLOR_BGR2RGB),
                        headset_img_rgb=cam2_rgb_image
                    )

            self.sheduler.adjust(idx, est_base_t_cam_and_points is not None)
            number_tries += 1
        self.debug_used_number_of_tries.append(number_tries)
        return est_base_t_cam_and_points


    def est_base_t_cam2_and_points(
            self,
            idx:int,
            cam2_rgb_image_features:list[Any],
            backward_transformations_names:list[tuple[str, Callable[[np.ndarray], np.ndarray]]],
            augmented_images:np.ndarray | None = None,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]] | None:
        """
        :return: None or base T_cam, points_image_1, points_image_2, world_obj_points, inliers
        """

        best_num_of_points = 0
        best_aug_option_name = ""
        best_image_points_cam1 = np.empty((0,2), dtype= np.float64)
        best_image_points_cam2 = np.empty((0,2), dtype= np.float64)
        best_world_obj_points = np.empty((0,3), dtype= np.float64)
        augmented_image_point_4_vis = None


        for i, (features_cam2, (name, backward)) in enumerate(zip(cam2_rgb_image_features, backward_transformations_names)):

            image_points_cam1, image_points_cam2 = self.extract_and_match.match_features(
                features1=self.cam1_features[idx],
                features2=features_cam2
            )

            if image_points_cam1.shape[0] < max(self.ransac_config.min_number_inlier_afterwards,6):
                continue

            if image_points_cam1.shape[0] < best_num_of_points:
                    continue

            image_points_cam1 = np.round(image_points_cam1)
            image_points_cam1[:, 0] = np.clip(image_points_cam1[:, 0], a_min=0, a_max=self.cam1_xyz_images[idx].shape[1]-1)
            image_points_cam1[:, 1] = np.clip(image_points_cam1[:, 1], a_min=0, a_max=self.cam1_xyz_images[idx].shape[0]-1)
            world_obj_points = np.array([self.cam1_xyz_images[idx][int(y),int(x)] for x,y in image_points_cam1])
            valid_points_mask = np.isfinite(world_obj_points).all(axis=1)

            if np.sum(valid_points_mask) > best_num_of_points:
                best_num_of_points = np.sum(valid_points_mask)
                best_aug_option_name = name
                best_image_points_cam1 = image_points_cam1[valid_points_mask]
                best_image_points_cam2 = backward(image_points_cam2)[valid_points_mask]
                best_world_obj_points = world_obj_points[valid_points_mask]

                if self.display_matching and augmented_images is not None:
                    augmented_image_point_4_vis = (augmented_images[i], image_points_cam2[valid_points_mask])


        if best_num_of_points < max(self.ransac_config.min_number_inlier_afterwards,6):
            return None

        self.debug_chosen_augmentations[best_aug_option_name] += 1

        cam2_t_base__inliers = estimate_point_pose_ransac(
            world_points=best_world_obj_points,
            img_points=best_image_points_cam2,
            intrinsic_matrix=self.cam2_mtx,
            config=self.ransac_config,
        )

        if cam2_t_base__inliers is None:
            return None
        
        self.debug_number_inliers.append(len(cam2_t_base__inliers[1]))
        
        if self.display_matching and augmented_image_point_4_vis is not None and self.cam1_bgr_images_for_vis is not None:
            aug_img, points2 = augmented_image_point_4_vis
            ExtractAndMatch.plot_matched_points(
                img1_rgb=cv2.cvtColor(self.cam1_bgr_images_for_vis[idx], code=cv2.COLOR_BGR2RGB),
                points1=best_image_points_cam1,
                img2_rgb=aug_img,
                points2=points2
            )

        return np.linalg.inv(cam2_t_base__inliers[0]), best_image_points_cam1, best_image_points_cam2, best_world_obj_points, cam2_t_base__inliers[1]
    

    # Plotting / debugging

    def get_sheduler(self)->Scheduler:
        return self.sheduler
    
    def get_number_chosen_augmenations(self)->dict[str, int]:
        return self.debug_chosen_augmentations
    
    def get_avg_number_of_tries(self)->SupportsFloat | None:
        return np.mean(self.debug_used_number_of_tries) if len(self.debug_used_number_of_tries) > 0 else None
    
    def get_avg_number_of_inliers(self)->SupportsFloat | None:
        return np.mean(self.debug_number_inliers) if len(self.debug_number_inliers) > 0 else None

    def print_used_augmentations(self)->None:
        print(f"Chosen augmentations:")
        for name, count in sorted(self.debug_chosen_augmentations.items(), key=lambda x: -x[1]):
            print(f" {name:<40}  {count}")
    
    def avg_number_of_tries(self)->SupportsFloat | None:
        return np.mean(self.debug_used_number_of_tries) if len(self.debug_used_number_of_tries) > 0 else None