import numpy as np
from dataclasses import dataclass
from typing import Literal
import torch
from PIL import Image
import cv2
from abc import ABC, abstractmethod


class Segmenter(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def get_object_masks(self, bgr_image:np.ndarray, visualize:bool = True)->np.ndarray:
        """
        :param rgb_image: HxWx3-uint8 rgb image
        :param visualize: If yes the segmented objects will be visualized
        :return: NxHxW boolean numpy array of the object masks
        """
        pass


@dataclass(frozen=True, kw_only=True)
class Sam3Prompt:
    text: str = "medium sized object"
    mask_threshold: float = 0.5
    threshold: float = 0.5

class SAM3Segmenter(Segmenter):
    _device = None
    _model = None
    _processor = None

    def __init__(self, prompt:Sam3Prompt) -> None:
        self.prompt = prompt

        if SAM3Segmenter._model is None:
            from transformers import Sam3Processor, Sam3Model
            SAM3Segmenter._device = "cuda" if torch.cuda.is_available() else "cpu"
            SAM3Segmenter._model = Sam3Model.from_pretrained("facebook/sam3").to(SAM3Segmenter._device)
            SAM3Segmenter._model.eval()
            SAM3Segmenter._processor = Sam3Processor.from_pretrained("facebook/sam3")


    def update_current_prompt(self, new_prompt:Sam3Prompt):
        self.prompt = new_prompt


    def get_object_masks(self, bgr_image:np.ndarray, visualize:bool = True)->np.ndarray:   

        if bgr_image.dtype in [np.float16, np.float32, np.float64]:
            bgr_image = (bgr_image * 255).astype(np.uint8)
        pil_rgb_image = Image.fromarray(cv2.cvtColor(np.clip(bgr_image, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB))

        inputs = SAM3Segmenter._processor(
            images=pil_rgb_image,
            text=self.prompt.text,
            return_tensors="pt"
        ).to(SAM3Segmenter._device)

        with torch.no_grad():
            outputs = SAM3Segmenter._model(**inputs)

            results = SAM3Segmenter._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.prompt.threshold,
                mask_threshold=self.prompt.mask_threshold,
                target_sizes=inputs.get("original_sizes").tolist()
            )[0]

            masks = np.empty((0, pil_rgb_image.height, pil_rgb_image.width), dtype=bool)
            if 'masks' in results and len(results['masks']) > 0:
                masks = results['masks'].detach().cpu().numpy()

            if visualize:
                display_image_masks(bgr_img=bgr_image, masks=masks)

            return masks


# TODO force only one in memory at a time
class YOLOv26Segmenter(Segmenter):
    def __init__(
            self,
            model:Literal["yoloe-26l-seg.pt", "yoloe-26s-seg.pt", "yoloe-26n-seg.pt", "yoloe-26m-seg.pt", "yoloe-26x-seg.pt"] = "yoloe-26s-seg.pt",
            prompts:list[str] = [
                "brick", "pen", "sphere", "round object", "tool", "toy", "plastic object", "metal object", "duplo", "lego", "pen"
            ],
        ) -> None:
        from ultralytics import YOLO

        from ultralytics.utils import LOGGER
        import logging

        LOGGER.setLevel(logging.ERROR)

        self.model = YOLO(model)
        self.model.set_classes(prompts)
    
    def get_object_masks(self, bgr_image:np.ndarray, visualize:bool = False)->np.ndarray:
        results = self.model.predict(
            cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB),
            imgsz = max(bgr_image.shape[0], bgr_image.shape[1])
        )[0]

        if results.masks is None:
            return np.empty((0, bgr_image.shape[0], bgr_image.shape[1]), dtype=bool)

        masks_np = results.masks.data.cpu().numpy().astype(bool)
        
        if bgr_image.shape[:2] != masks_np.shape[1:]:
            masks_np = np.stack([
                cv2.resize(
                    mask.astype(np.uint8),
                    (bgr_image.shape[1], bgr_image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                for mask in masks_np
            ])

        if visualize:
            display_image_masks(bgr_img=bgr_image, masks=masks_np)


        return masks_np




def display_image_masks(bgr_img:np.ndarray, masks:np.ndarray):
    """
    Visualize objects masks on an image
    :param bgr_img: a HxWx3-uint8 BGR image
    :param masks: NxHxW-bool masks of objedts in that image
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(rgb_img.astype(int))


    mask_colors = plt.cm.jet(np.linspace(0, 1, masks.shape[0]))
    overlay = np.zeros_like(rgb_img, dtype = np.float32)
    for i, mask in enumerate(masks):
        overlay[mask > 0] += mask_colors[i, :3] * 255
    overlay = np.clip(overlay, 0, 255)
    axes[0].imshow(overlay.astype(int), alpha = 0.4)
    axes[1].imshow(overlay.astype(int))

    plt.show()