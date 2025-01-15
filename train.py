import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import clip
from tqdm import tqdm

from demo.data import CustomCompose
from demo.simmim import MaskGenerator
from demo.data import MyCocoDetection
from demo.image_encoder import ResnetEncoder
from demo.detection import FCOSDetector
from demo.reconstruction import Recostruction
from demo.loss import ReconstructionLoss, TextureConsistencyLoss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


"""データ処理"""
# 前処理+データ拡張
transform = CustomCompose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                                transforms.Resize((512, 512))])

# データセットの読み込み
dataset = MyCocoDetection(root="dataset/detection", annFile="dataset/detection/train_quadrant_enumeration_fdi.json", transforms=transform)
# データローダーの設定
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn = lambda x: tuple(zip(*x)))


"""モデルの読み込み"""
# 画像エンコーダー、検出器、再構築器の読み込み
encoder = ResnetEncoder()   
detector = FCOSDetector()
recostruction = Recostruction(encoder_outchannels=256)

# 損失関数
reconstruction_loss = ReconstructionLoss()
texture_consistency_loss = TextureConsistencyLoss()

# 最適化アルゴリズム
optimizer = optim.Adam(list(encoder.parameters()) + list(detector.parameters()) + list(recostruction.parameters()), lr=1e-4)

"""学習の実行"""
# ハイパーパラメータ
NUM_EPOCHS = 1000
lr = 1e-4


# モデルをcudaに移動
encoder.to(device)
detector.to(device)
recostruction.to(device)

# 学習ループ
for epoch in range(NUM_EPOCHS):
    encoder.train()
    detector.train()
    recostruction.train()

    for images, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", unit="batch"):
        images = [image.to(device) for image in images]
        targets = [{k: v.to(device) for k, v in target.items()} for target in targets]


        # imagesをB,C,H,Wに変換
        images = images.stack(images, dim=0)

        # マスク生成
        mask_batch_size = images.shape[0]
        mask_generator = MaskGenerator(batch_size=mask_batch_size)

        # imagesにマスクをかける
        masks = mask_generator()
        # 画像の形状に合わせてマスクをリシェイプ
        masks = masks.unsqueeze(1)
        # マスクをcudaに移動
        masks = masks.to(device)

        masked_images = images * (1 - masks)

        # マスクをかけた画像をencoderに通す
        features = encoder.forward(masked_images)
        # そのうち最も浅い特徴マップを取得
        shallow_feature = features['0']

        # 特徴量抽出された画像を復元
        reconstructed_image = recostruction.forward(shallow_feature)    

        # 再構築ブランチの損失関数を計算
        reconstruction_loss = ReconstructionLoss(images, reconstructed_image)
        reconstruction_loss = reconstruction_loss.calculate_loss()

        # テクスチャー一貫性ブランチの損失を計算
        texture_consistency_loss = TextureConsistencyLoss(images, reconstructed_image)
        texture_consistency_loss = texture_consistency_loss.calculate_loss()

        # 検出ブランチの損失関数を計算
        detection_loss = detector.forward(images, targets)
        detection_loss = sum(loss for loss in detection_loss.values())

        # 総損失を計算
        total_loss = reconstruction_loss + texture_consistency_loss + detection_loss

        # 勾配を計算
        optimizer.zero_grad()
        total_loss.backward()
        # パラメータ更新
        optimizer.step()

    # 1エポック終了時のロスを表示
    print(f"Epoch {epoch+1}, Reconstruction Loss: {reconstruction_loss.item()}, Texture Consistency Loss: {texture_consistency_loss.item()}, Detection Loss: {detection_loss.item()}, Total Loss: {total_loss.item()}")
    # 10エポックごとにモデルを保存
    if (epoch+1) % 10 == 0:
        torch.save(encoder.state_dict(), f"encoder_{epoch+1}.pth")
        torch.save(detector.state_dict(), f"detector_{epoch+1}.pth")
        torch.save(recostruction.state_dict(), f"recostruction_{epoch+1}.pth")









