import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import clip


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
        original_image = self.clip_preprocess(self.original_image).unsqueeze(0).to(self.device)
        reconstructed_image = self.clip_preprocess(self.reconstructed_image).unsqueeze(0).to(self.device)
        # 512の長さのベクトルにエンコード
        with torch.no_grad():
            original_image_features = self.clip_model.encode_image(original_image)
            reconstructed_image_features = self.clip_model.encode_image(reconstructed_image)
        # 損失の計算
        original_image_features = original_image_features / original_image_features.norm(dim=-1, keepdim=True)
        reconstructed_image_features = reconstructed_image_features / reconstructed_image_features.norm(dim=-1, keepdim=True)
        cosine_sim_loss = F.cosine_similarity(original_image_features, reconstructed_image_features, dim=-1)
        loss = 1 - cosine_sim_loss
        
        return loss



    def calculate_loss(self):
        reconstruction_loss = ReconstructionLoss(self.device, self.original_image, self.reconstructed_image)
        texture_consistency_loss = TextureConsistencyLoss(self.device, self.original_image, self.reconstructed_image)

        return reconstruction_loss + texture_consistency_loss






