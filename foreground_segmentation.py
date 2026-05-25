import numpy as np
from dataclasses import dataclass, field
from transformers import Sam3Processor, Sam3Model
import torch
from tqdm import tqdm
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from shared_utilities import get_image_type_hxw, assert_mxnx3_np_uint8_image


@dataclass(frozen=True, kw_only=True)
class ForegroundSegmentationConfig:
    """
    Sets the parameters for an RANSAAC 3d pose estimation.
    :param threshold: certainty needed by sam3 to detect an object
    :param mask_threshold certainty for mask generation by sam3
    :param prompts: A list of (sam3prompt, score) tuples, the scores over the different prompts will be added. 
    :param min_score: The score an pixel has to reach, to be considered foreground
    """
    threshold_fg:float = 0.5
    threshold_bg:float = 0.3
    mask_threshold_fg:float = 0.5
    mask_threshhold_bg:float = 0.3

    prompts:list[tuple[str, float]] = field(default_factory= lambda:
        [("distinct objects", 1),("foreground", 1),("tabletop", -1),("background", -1),("big plain surfaces", -2),("white paper", -2)]
    )
    min_score:float = 0

    def __post_init__(self):
        assert 0 <= self.threshold_fg <= 1.0, f"Threshold {self.threshold_fg} not in [0,1]"
        assert 0 <= self.mask_threshold_fg <= 1.0, f"Mask threshhold {self.mask_threshold_fg} not in [0,1]"
        assert 0 <= self.threshold_fg <= 1.0, f"Threshold {self.threshold_fg} not in [0,1]"
        assert 0 <= self.mask_threshold_fg <= 1.0, f"Mask threshhold {self.mask_threshold_fg} not in [0,1]"

foreground_segmentation_config_standard = ForegroundSegmentationConfig()
foreground_segmentation_config_hard = ForegroundSegmentationConfig(
    threshold=0.8, 
    mask_threshold=0.8,
    min_score=0.9
)
foreground_segmentation_config_and = ForegroundSegmentationConfig(
    threshold=0.8, 
    mask_threshold=0.9,
    min_score=1.9
)

def create_foreground_masks(
        images:np.ndarray,
        config: ForegroundSegmentationConfig = foreground_segmentation_config_standard,
        visualize_masks:bool = False,
    ) -> np.ndarray:
    """
    Uses Sam3 to detect objects/the foreground and returns a mask for each image, which is `True` where an object was detected
    :param images: NxHxWx3-uint8/float16/float32/float64 numpy array for the images (BGR)
    :param config: how to segment the images
    :param visualize_masks: Whether to visualize the masks for debugging
    :return: NxHxW boolean numpy array of the foreground masks
    """
    _ = all([assert_mxnx3_np_uint8_image(img=img) for img in images])

    print(f"generating foreground masks for {images.shape[0]} {get_image_type_hxw(images[0])}images")
    if images.dtype in [np.float16,np.float32, np.float64]:
        images = (images*255).astype(np.uint8)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Sam3Model.from_pretrained("facebook/sam3").to(device)
    processor = Sam3Processor.from_pretrained("facebook/sam3")

    masks = []
    for image in tqdm(images):
        pil_rgb_image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=float)
        for text, score in config.prompts:
            inputs = processor(images=pil_rgb_image, text = text,return_tensors="pt").to(device)
            
            with torch.inference_mode():
                outputs = model(**inputs)

                results = processor.post_process_instance_segmentation(
                    outputs,
                    threshold=config.threshold,
                    mask_threshold=config.mask_threshold,
                    target_sizes=inputs.get("original_sizes").tolist()
                )[0]

            if 'masks' in results and len(results['masks']) > 0:
                instance_masks = results['masks'].cpu().numpy()
                aggregated_mask = np.any(instance_masks, axis=0)
                mask = mask + score * aggregated_mask.astype(float)

        masks.append(mask > config.min_score)
        if visualize_masks:
            plt.figure(figsize=(15, 5))
            plt.imshow(pil_rgb_image)
            plt.imshow(masks[-1], alpha=0.5, cmap='jet')
            plt.show()
    
    return np.array(masks)