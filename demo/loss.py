import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import clip
from torchvision.transforms import Resize, Normalize


class ReconstructionLoss(nn.Module):
    def __init__(self, original_image, reconstructed_image, device="cuda"):
        super().__init__()
        self.device = device
        self.original_image = original_image
        self.reconstructed_image = reconstructed_image

    def calculate_loss(self, type="l1"):
        if type == "l1":
            loss = nn.L1Loss()
        elif type == "l2":
            loss = nn.MSELoss()
        else:
            raise ValueError(f"Invalid loss type: {type}")
        
        return loss(self.original_image, self.reconstructed_image)
    
class TextureConsistencyLoss(nn.Module):
    def __init__(self, original_image, reconstructed_image, device="cuda"):
        super().__init__()
        self.device = device
        self.original_image = original_image
        self.reconstructed_image = reconstructed_image

        # CLIPモデルの読み込み
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/16", device=self.device)

    def calculate_loss(self):
        # 画像の前処理
        """
        original_image = self.clip_preprocess(self.original_image).to(self.device)
        reconstructed_image = self.clip_preprocess(self.reconstructed_image).to(self.device)
        """
        ## リサイズ、正規化、チャネル数を3に変換
        original_image = Resize(224)(self.original_image)
        original_image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(original_image)

        reconstructed_image = Resize(224)(self.original_image)
        reconstructed_image = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(reconstructed_image)
        
        # 512の長さのベクトルにエンコード
        with torch.no_grad():
            original_image_features = self.clip_model.encode_image(original_image)
            reconstructed_image_features = self.clip_model.encode_image(reconstructed_image)
        # 損失の計算
        original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)
        reconstructed_image_features = reconstructed_image_features / reconstructed_image_features.norm(dim=-1, keepdim=True)
        cosine_sim_loss = F.cosine_similarity(original_image_features, reconstructed_image_features, dim=-1)
        losses = 1 - cosine_sim_loss
        loss = sum(loss for loss in losses)
        
        
        
        
        
        return loss









