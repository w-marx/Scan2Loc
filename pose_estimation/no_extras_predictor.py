import cv2
import numpy as np
from predictor_handling import *
from extractors_and_matchers import *

class NoExtrasPredictor(PosePredictor):
    def __init__(
            self,
            cam2_mtx:np.ndarray,
            #extract_and_match:ExtractAndMatch = ExtractAndLightGlue(extractor="SuperPoint"),
            extract_and_match:ExtractAndMatch = ExtractAndMatchLoMa(),
            use_rotation_augmentations:bool = False,
            crop_augmentations:list[float] | None = None,
            min_number_inlier:int = 6,
            ransac_itterations:int = 10000,
            ransac_reprojection_error:float = 5.0,
            ransac_confidence:float = 0.99
        ):
        super().__init__()
        self.cam2_mtx = cam2_mtx
        self.extract_and_match = extract_and_match
        self.min_number_inlier = min_number_inlier
        self.ransac_itterations = ransac_itterations
        self.ransac_reprojection_error = ransac_reprojection_error
        self.ransac_confidence = ransac_confidence

        self.rotation_augmentations = [Augmentation()]
        if use_rotation_augmentations:
            self.rotation_augmentations.append(Rotate180Deg())
        
        self.crop_augmentations = [Augmentation()]
        if crop_augmentations is not None:
            self.crop_augmentations + [CropImage(x) for x in crop_augmentations if 0 <= x < 1.0]  

        self.use_rotation_augmentations = use_rotation_augmentations

    def est_base_t_cam2(self,
                        cam1_bgr_image:np.ndarray,
                        base_xyz_image:np.ndarray,
                        cam2_bgr_image: np.ndarray,
                        point_cloud:np.ndarray,
                        ) -> np.ndarray | None:

  
        
        augmentation_options_names = []
        world_obj_points_options = []
        image_points_cam2_options = []

        for c_aug in self.crop_augmentations:
            for r_aug in self.rotation_augmentations:
                augmented_image, backward_aug2 = c_aug.forward(cam2_bgr_image)
                augmented_image, backward_aug1 = r_aug.forward(augmented_image)

                image_points_cam1, image_points_cam2 = self.extract_and_match.get_matched_points(
                    cv2.cvtColor(cam1_bgr_image, cv2.COLOR_BGR2RGB),
                    cv2.cvtColor(augmented_image, cv2.COLOR_BGR2RGB)
                )

                world_obj_points = np.array([base_xyz_image[int(np.round(y)),int(np.round(x))] for x,y in image_points_cam1])
                if world_obj_points.shape[0] > 5:
                    augmentation_options_names.append(f"{c_aug}x{r_aug}")
                    world_obj_points_options.append(world_obj_points)
                    image_points_cam2_options.append(backward_aug2(backward_aug1(image_points_cam2)))
        
        if len(world_obj_points_options) < 1:
            return None
    
        best_option_idx = np.argmax(np.array([x.shape[0] for x in world_obj_points_options]))
        #print(f"chosen rot aug.: {augmentation_options_names[best_option_idx]}")
        world_obj_points = world_obj_points_options[best_option_idx]
        image_points_cam2 = image_points_cam2_options[best_option_idx]


        success, r_img_t_obj, t_img_t_obj, inliers = cv2.solvePnPRansac(
            world_obj_points, image_points_cam2, self.cam2_mtx, None,
            iterationsCount = self.ransac_itterations,
            reprojectionError=self.ransac_reprojection_error,
            confidence = self.ransac_confidence,
            flags = cv2.SOLVEPNP_EPNP
        )
        if not success or len(inliers) < self.min_number_inlier:
           return None
        
        cam2_t_base = np.eye(4)
        cam2_t_base[:3, :3] = cv2.Rodrigues(r_img_t_obj)[0]
        cam2_t_base[:3, 3] = t_img_t_obj.flatten()

        return np.linalg.inv(cam2_t_base)




if __name__ == "__main__":
    data = PredictionData.from_folder("/home/wmarx/AR-Headset-Localization-in-Robot-Scanned-Workspaces-A-Benchmark-Pipeline/data_preprocessing/out_data")
    predictor = NoExtrasPredictor(data.headset_intrinsics, extract_and_match=ExtractAndMatchLoMa(loma_variant="LoMaB"))
    grader = OnePredictorOneDatasetGrader(predictor=predictor, data=data)
    
    grader.visualize_predictions()
    
    print(f"median rot error: {np.rad2deg(grader.median_rotational_error())} degrees")
    print(f"median translational error: {grader.median_translational_error()*1000} mm")
    