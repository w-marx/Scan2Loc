import numpy as np
from dataclasses import dataclass, field
import torch
from PIL import Image
import cv2

def log_memory_usage(stage=""):
    if torch.cuda.is_available():
        print(f"{stage} - GPU: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
    import psutil
    print(f"{stage} - RAM: {psutil.Process().memory_info().rss/1024**3:.2f} GB")

def use_sam3_on_image_to_get_masks(
        pil_rgb_image,
        mask_threshold:float,
        threshold:float,
        prompt:str
):

    if not hasattr(use_sam3_on_image_to_get_masks, 'model'):
        from transformers import Sam3Processor, Sam3Model
        use_sam3_on_image_to_get_masks.device = "cuda" if torch.cuda.is_available() else "cpu"
        use_sam3_on_image_to_get_masks.model = Sam3Model.from_pretrained("facebook/sam3").to(use_sam3_on_image_to_get_masks.device)
        use_sam3_on_image_to_get_masks.processor = Sam3Processor.from_pretrained("facebook/sam3")

    inputs = use_sam3_on_image_to_get_masks.processor(
        images=pil_rgb_image,
        text=prompt,
        return_tensors="pt"
    ).to(use_sam3_on_image_to_get_masks.device)

    with torch.inference_mode():
        outputs = use_sam3_on_image_to_get_masks.model(**inputs)

        results = use_sam3_on_image_to_get_masks.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=inputs.get("original_sizes").tolist()
        )[0]

        del inputs
        del outputs

        masks = np.empty((0, pil_rgb_image.height, pil_rgb_image.width), dtype=bool)
        if 'masks' in results and len(results['masks']) > 0:
            masks = results['masks'].cpu().numpy()
        del results
        return masks

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
    log_memory_usage("started new get_object_masks")

    if image.dtype in [np.float16, np.float32, np.float64]:
        image = (image * 255).astype(np.uint8)
    pil_rgb_image = Image.fromarray(cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2RGB))

    masks = use_sam3_on_image_to_get_masks(
        pil_rgb_image=pil_rgb_image,
        prompt=prompt.text,
        mask_threshold=prompt.mask_threshold,
        threshold=prompt.threshold
    )

    log_memory_usage("ended get_object_masks")

    return masks