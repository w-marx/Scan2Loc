import numpy as np
from dataclasses import dataclass, field
import torch
from PIL import Image
import cv2

def use_sam3_on_image(
        pil_rgb_image,
        mask_threshold:float,
        threshold:float,
        prompt:str
):

    if not hasattr(use_sam3_on_image, 'model'):
        from transformers import Sam3Processor, Sam3Model
        use_sam3_on_image.device = "cuda" if torch.cuda.is_available() else "cpu"
        use_sam3_on_image.model = Sam3Model.from_pretrained("facebook/sam3").to(use_sam3_on_image.device)
        use_sam3_on_image.processor = Sam3Processor.from_pretrained("facebook/sam3")

    inputs = use_sam3_on_image.processor(
        images=pil_rgb_image,
        text=prompt,
        return_tensors="pt"
    ).to(use_sam3_on_image.device)

    with torch.inference_mode():
        outputs = use_sam3_on_image.model(**inputs)

        results = use_sam3_on_image.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]
        return results

@dataclass(frozen=True, kw_only=True)
class Sam3Prompt:
    text: str = "medium sized object"
    mask_threshold: float = 0.5
    threshold: float = 0.5

def get_object_masks(
        image: np.ndarray,
        prompt:Sam3Prompt = Sam3Prompt()
) -> np.ndarray:
    """
    :param image: HxWx3-uint8/float16/float32/float64 for the image (BGR)
    :param prompt: how to segment the image
    :return: NxHxW boolean numpy array of the object masks
    """

    if image.dtype in [np.float16, np.float32, np.float64]:
        image = (image * 255).astype(np.uint8)
    pil_rgb_image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))

    results = use_sam3_on_image(
        pil_rgb_image=pil_rgb_image,
        prompt=prompt.text,
        mask_threshold=prompt.mask_threshold,
        threshold=prompt.threshold
    )

    objects = np.empty((0, image.shape[0], image.shape[1]), dtype=bool)
    if 'masks' in results and len(results['masks']) > 0:
        objects = results['masks'].cpu().numpy()
    return objects