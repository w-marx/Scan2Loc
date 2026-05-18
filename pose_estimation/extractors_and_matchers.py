import torch, cv2
from typing import Literal

class ExtractAndMatch:
    def get_matched_points(self,img1, img2):
        raise NotImplementedError("Not implemented in base class")

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
        }

        self.feature_name = extractor.lower()

        self._rbd = rbd
        self._numpy_image_to_torch = numpy_image_to_torch

        self.extractor = extractors[extractor](max_num_keypoints=max_num_keypoints).eval().cuda()
        self.matcher = LightGlue(features=self.feature_name).eval().cuda()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def get_matched_points(self, img1_rgb, img2_rgb):
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