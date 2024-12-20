import torch
import torch.nn as nn
import numpy as np


class MaskGenerator:
    def __init__(self, image,input_size=512, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.image = image
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio


        assert self.input_size % self.model_patch_size == 0, "input_size must be divisible by model_patch_size"
        assert self.input_size % self.mask_patch_size == 0, "input_size must be divisible by mask_patch_size"

        self.rand_size = self.input_size // self.model_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size

        self.token_count = self.rand_size **2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))

    def __call__(self):

        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1

        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)

        # 画像の形状に合わせてマスクをリシェイプ
        if len(self.image.shape) == 3:
            H, W, C = self.image.shape
            mask = np.expand_dims(mask, axis=-1).repeat(C, axis=-1)  # チャネル次元を追加
        else:
            B, H, W, C = self.image.shape
            mask = np.expand_dims(mask, axis=0).repeat(B, axis=0)  # バッチ次元を追加
            mask = np.expand_dims(mask, axis=-1).repeat(C, axis=-1)  # チャネル次元を追加

        # マスクをかける
        mask = torch.from_numpy(mask)
        image = self.image * (1 - mask)

        return image
    

