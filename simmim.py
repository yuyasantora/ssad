import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from model import SwinBackbone

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MaskGenerator:
    def __init__(self, input_size=512, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
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

        return mask

class MIMHead(nn.Module):
    """Masked Image Modeling Head"""
    def __init__(self, backbone, encorder_stride=32):
        super().__init__()
        self.backbone = backbone
        self.encorder_stride = encorder_stride
        self.mask_generator = MaskGenerator()

        # デコーダー層の定義
        self.decoder_embed = nn.Linear(self.backbone.head.out_features, 512)
        self.decoder_blocks = nn.ModuleList([
            nn.TransformerDecoderLayer(d_model=512, nhead=8)
            for _ in range(4)
        ])
        self.decoder_norm = nn.LayerNorm(512)
        self.decoder_pred = nn.Linear(512, self.encorder_stride**2)

    def forward(self, x):
        # マスク画像の生成
        mask = self.mask_generator()
        mask = torch.from_numpy(mask).to(device)

        # 画像のマスキング
        if len(x.shape) == 3:
            C,H,W = x.shape
        
        if len(x.shape) == 4:
            B,C,H,W = x.shape
        

        mask = mask.view(1,H,W,1)
        x_masked = x * (1- mask)

        # エンコーダーの特徴量の取得
        features = self.backbone(x_masked)
        encoded = features[-1]

        # デコーダ
        x = self.decoder_embed(encoded)
        for block in self.decoder_blocks:
            x = block(x, encoded)
        x = self.decode_norm(x)
        x = self.decoder_pred(x)

        # 予測結果の整形
        patches = x.reshape(B, H//self.encorder_stride, W//self.encorder_stride, self.encorder_stride**2)
        patches = patches.permute(0, 3, 1, 2)
        reconstructed = F.fold(
            patches,
            output_size=(H, W),
            kernel_size=self.encorder_stride,
            stride=self.encorder_stride,
        )
        return reconstructed, mask
    
class SwinWithMIM(nn.Module):
    """MIMを含むSwin Transformer"""
    def __init__(self, backbone):
        super().__init__()
        # .model.SwinBackboneをインスタンス化
        self.backbone = backbone
        # MIMHeadをインスタンス化
        self.mim_head = MIMHead(self.backbone)

    # 事前学習
    def forward(self, x):
        reconstructed, mask = self.mim_head(x)

        # 再構築損失の計算 (コサイン類似度)
        loss_theta = 1- F.cosine_similarity(x, reconstructed, dim=1)

        # Texture Consistency Loss (L2ノルム)
        loss_d = F.mse_loss(x, reconstructed)

        

        # 総損失の計算
        loss = loss_theta + loss_d 

        return loss

















