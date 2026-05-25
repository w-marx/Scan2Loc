import numpy as np
from typing import Callable
import cv2

class Augmentation:
    """
    Class of image augmentations, that provide a way of modifying and image.
    forward will return the image and a function to map image coordinages from the new
    image coordinates to the old ones
    """
    def __str__(self) -> str: return "Identity"
    def forward(self,img:np.ndarray)->tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        return img, lambda x:x

class Rotate180Deg(Augmentation):
    def __str__(self)-> str: return "Rotate 180°"
    def forward(self,img:np.ndarray)->tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        w, h = img.shape[:2]
        img = cv2.rotate(img, cv2.ROTATE_180)
        backward = lambda img_points: np.array([[w-x, h-y] for x,y in img_points])
        return img, backward

class CropImage(Augmentation):
    def __init__(self, relative_crop_amount:float = 0):
        self.relative_crop_amount = relative_crop_amount
    
    def __str__(self)->str: return f"CropImage {self.relative_crop_amount}"

    def forward(self,img:np.ndarray)->tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
        crop_pixels = int(self.relative_crop_amount * np.min(img.shape[:2])/2)
        img = img[crop_pixels:-crop_pixels,crop_pixels:-crop_pixels,:]
        backward = lambda img_points: img_points + crop_pixels
        return img, backward