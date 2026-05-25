from typing import Literal, Callable
from PIL import Image
import numpy as np
import cv2, sys, os, torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from shared_utilities import *

import matplotlib.pyplot as plt

class ExtractAndMatch:
    def get_matched_points(self,img1_rgb:np.ndarray, img2_rgb:np.ndarray, plot_results:bool = False)->tuple[np.ndarray, np.ndarray]:
        """
        :param img1_rgb: An RGB image as HxWx3-uint8 numpy array
        :param img2_rgb: An RGB image as HxWx3-uint8 numpy array
        :param plot_results: wheather to plot the matched features
        :return: a tuple of image Points as 2 Nx2 numpy arrays (in the x-y format)
        """
        raise NotImplementedError("Not implemented in base class")
    
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
    
    def get_matched_points(self,img1_rgb:np.ndarray, img2_rgb:np.ndarray, plot_results:bool = False)->tuple[np.ndarray, np.ndarray]:
        _ = assert_mxnx3_np_uint8_image(img1_rgb)
        _ = assert_mxnx3_np_uint8_image(img2_rgb)

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

        if plot_results:
            self.plot_matched_points(
                img1_rgb=img1_rgb, img2_rgb=img2_rgb, points1=image_points_cam1_cpu_np, points2=image_points_cam2_cpu_np
            )

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

    def get_matched_points(self,img1_rgb:np.ndarray, img2_rgb:np.ndarray, plot_results:bool = False)->tuple[np.ndarray, np.ndarray]:
        _ = assert_mxnx3_np_uint8_image(img1_rgb)
        _ = assert_mxnx3_np_uint8_image(img2_rgb)

        h1, w1 = (img1_rgb.shape[0] // 14)*14, (img1_rgb.shape[1] // 14)*14
        img1_rgb_m14 = img1_rgb[:h1, : w1, :]

        h2, w2 = (img2_rgb.shape[0] // 14)*14, (img2_rgb.shape[1] // 14)*14
        img2_rgb_m14 = img2_rgb[:h2, : w2, :]
    
        img1_tensor = torch.from_numpy(img1_rgb_m14).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img2_tensor = torch.from_numpy(img2_rgb_m14).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        kpts1, kpts2 = self.model.match(img1_tensor, img2_tensor)

        if plot_results:
            self.plot_matched_points(
                img1_rgb=img1_rgb, img2_rgb=img2_rgb, points1=kpts1, points2=kpts2
            )
        
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

    def get_matched_points(self,img1_rgb:np.ndarray, img2_rgb:np.ndarray, plot_results:bool = False)->tuple[np.ndarray, np.ndarray]:
        _ = assert_mxnx3_np_uint8_image(img1_rgb)
        _ = assert_mxnx3_np_uint8_image(img2_rgb)
    
        pil_img1 = Image.fromarray(img1_rgb)
        pil_img2 = Image.fromarray(img2_rgb)

        inputs = self.processor([pil_img1, pil_img2], return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**inputs)

        image_sizes = [[(pil_img1.height, pil_img1.width),(pil_img2.height, pil_img2.width)]]
        output = self.processor.post_process_keypoint_matching(outputs, image_sizes, threshold=self.matching_threshhold)[0]

        kpts1 = output["keypoints0"].cpu().numpy().astype(np.float32)
        kpts2 = output["keypoints1"].cpu().numpy().astype(np.float32)

        if plot_results:
            self.plot_matched_points(
                img1_rgb=img1_rgb, img2_rgb=img2_rgb, points1=kpts1, points2=kpts2
            )
        
        return kpts1, kpts2