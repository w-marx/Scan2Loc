from typing import Literal, SupportsFloat
from PIL import Image
import numpy as np
import torch
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from shared.assertion_helpers import assert_mxnx3_np_uint8_image


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
    def plot_matched_points(img1_rgb:np.ndarray, img2_rgb:np.ndarray, points1:np.ndarray, points2:np.ndarray, ax:Axes|None = None):
        """
        :param img1_rgb: An RGB image as HxWx3-uint8 numpy array
        :param img2_rgb: An RGB image as HxWx3-uint8 numpy array
        :param points1: Nx2 array of 2d points of the form [[x1,y1], ...] in img1_rgb points1[i] is matched to points2[i]
        :param points1: Nx2 array of 2d points
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

        if ax is None:
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
    def __init__(
            self, 
            loma_variant:Literal["LoMaB", "LoMaB128", "LoMaL", "LoMaG", "LoMaR"] = "LoMaG", 
            num_keypoints:int | None = None, 
            filter_threshold:float | None = None
        ):
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

        self.num_keypoints = num_keypoints
        self.filter_threshold = filter_threshold

    def get_features(self, img_rgb:np.ndarray):
        """
        :param img_rgb: An HxWx3-unint8 RGB image
        :return: an torch tensor of the image with height & width %14 = 0
        """
        assert assert_mxnx3_np_uint8_image(img_rgb)

        h1, w1 = (img_rgb.shape[0] // 14)*14, (img_rgb.shape[1] // 14)*14
        img_rgb_m14 = img_rgb[:h1, : w1, :]
        img_tensor = torch.from_numpy(img_rgb_m14).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        keypoints_A, descriptors_A, h1, w1 = self.model.detect_and_describe(
            img_tensor, self.num_keypoints
        )

        return (keypoints_A, descriptors_A, h1, w1)
    
    
    @staticmethod
    def _to_pixel_coords(flow, h1, w1):
        """
        Method copied from: https://github.com/davnords/LoMa/blob/main/src/loma/loma.py
        """
        flow = torch.stack(
            (
                w1 * (flow[..., 0] + 1) / 2,
                h1 * (flow[..., 1] + 1) / 2,
            ),
            dim=-1,
        )
        return flow


    @staticmethod
    def _filter_matches(scores: torch.Tensor, th: float):
        """
        Method copied from: https://github.com/davnords/LoMa/blob/main/src/loma/loma.py
        """
        max0, max1 = scores.max(2), scores.max(1)
        m0, m1 = max0.indices, max1.indices
        indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
        indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
        mutual0 = indices0 == m1.gather(1, m0)
        mutual1 = indices1 == m0.gather(1, m1)
        mscores0 = torch.where(mutual0, max0.values, max0.values.new_tensor(0))
        mscores1 = torch.where(mutual1, mscores0.gather(1, m1), mscores0.new_tensor(0))
        valid0 = mutual0 & (mscores0 > th)
        valid1 = mutual1 & valid0.gather(1, m1)
        m0 = torch.where(valid0, m0, -1)
        m1 = torch.where(valid1, m1, -1)
        return m0, m1, mscores0, mscores1   
     

    def match_features(self, features1, features2) -> tuple[np.ndarray, np.ndarray]:
        """
        :param features1: A torch tensor of a image with height & width %14 = 0
        :param features2: Another torch tensor of a image with height & width %14 = 0
        :return: a tuple of image Points as 2 Nx2 numpy arrays (in the x-y format)
        """
        keypoints_A, descriptors_A, h1, w1 = features1
        keypoints_B, descriptors_B, h2, w2 = features2


        if self.filter_threshold is None:
            filter_threshold = self.model.cfg.filter_threshold

        scores = self.model(keypoints_A, keypoints_B, descriptors_A, descriptors_B)["scores"]
        m0, _, _, _ = self._filter_matches(scores, filter_threshold)

        valid = m0[0] > -1
        matched_A = keypoints_A[0][torch.where(valid)[0]]
        matched_B = keypoints_B[0][m0[0][valid]]

        return self._to_pixel_coords(matched_A, h1, w1).cpu().numpy(), self._to_pixel_coords(matched_B, h2, w2).cpu().numpy()


class ExtractAndMatchEffLoFTR(ExtractAndMatch):
    _processor = None
    _model = None

    def __init__(self, matching_threshhold:float = 0.3):
        """
        Extract and match based on efficient LoFTR
        :param matching_threshhold: filter threshhold int [0,1] for matchings
        """
        assert 0 <= matching_threshhold <= 1, f"invalid threshhold: {matching_threshhold} not in [0,1]"

        if ExtractAndMatchEffLoFTR._processor is None:
            import transformers
            from transformers import AutoImageProcessor, AutoModelForKeypointMatching
            ExtractAndMatchEffLoFTR._processor = AutoImageProcessor.from_pretrained("zju-community/efficientloftr") 
            ExtractAndMatchEffLoFTR._model = AutoModelForKeypointMatching.from_pretrained("zju-community/efficientloftr")
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
        inputs = ExtractAndMatchEffLoFTR._processor([features1, features2], return_tensors="pt")
        with torch.no_grad():
            outputs = ExtractAndMatchEffLoFTR._model(**inputs)

        image_sizes = [[(features1.height, features1.width),(features2.height, features2.width)]]
        output = ExtractAndMatchEffLoFTR._processor.post_process_keypoint_matching(outputs, image_sizes, threshold=self.matching_threshhold)[0]

        kpts1 = output["keypoints0"].cpu().numpy().astype(np.float32)
        kpts2 = output["keypoints1"].cpu().numpy().astype(np.float32)
        return kpts1, kpts2