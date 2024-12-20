import torch
import torch.nn as nn
from torchvision.models.detection import fcos_resnet50_fpn

from demo.image_encoder import ResnetEncoder

class FCOSDetector(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        self.model = fcos_resnet50_fpn(pretrained=False)
        self.model.to(self.device)
        # バックボーンをImageEncoderと共有
        self.model.backbone = ResnetEncoder(self.device)

    def forward(self, image):
        return self.model(image)
    

