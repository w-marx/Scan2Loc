import numpy as np

class ImageMaskStorage:
    """
    Stores boolean masks memory efficient (1 bit per bool)
    """

    def __init__(self, height:int, width:int):
        self.h = height
        self.w = width
        self.storage = []
        self.id_storage = {}
        self.c_idx = 0

    def add_new_mask(self,mask:np.ndarray):
        """
        Adds the uint8/bool mask to storage
        """
        assert mask.shape == (self.h, self.w), f"got {mask.shape} instead of {self.h},{self.w}"

        if not np.issubdtype(mask.dtype, np.bool_):
            mask = mask != 0

        list_idx = int(self.c_idx/8)
        bit_idx = self.c_idx % 8

        if bit_idx == 0:
            self.storage.append(np.zeros((self.h, self.w), dtype = np.uint8))

        bitfield = np.left_shift(mask.astype(np.uint8), bit_idx).astype(np.uint8)
        self.storage[list_idx] = np.bitwise_or(self.storage[list_idx], bitfield)

        self.c_idx += 1
    
    def see_mask(self, idx:int)->np.ndarray:
        assert idx < self.c_idx
        list_idx = idx // 8
        bit_idx = idx % 8
        return ((self.storage[list_idx] >> bit_idx) & 1).astype(bool)