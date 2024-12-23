import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

"""データセットの読み込み"""
# 前処理をするクラス
class PreCompose(object):
    def __init__(self, transforms:list):
        self.transforms = transforms.Compose(transforms)
    
    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
            return img

precompose = PreCompose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), transforms.Resize((512, 512))])

# データセットの読み込み
## データセットのクラス定義
class UnlabelledDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.image_files = [image_file for image_file in os.listdir(root)]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_path = os.path.join(self.root, self.image_files[index])
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (1024, 1024))
        image = precompose(image)

        return image

## データセットクラスをオブジェクト化
dataset = UnlabelledDataset(root="./xrays")
dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=4)
