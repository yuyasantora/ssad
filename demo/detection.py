import torch
import torch.nn as nn
from torchvision.models.detection import fcos_resnet50_fpn

from demo.image_encoder import ResnetEncoder

class FCOSDetector(nn.Module):
    def __init__(self, device="cuda",num_classes=32):
        super().__init__()
        self.device = device
        self.model = fcos_resnet50_fpn(pretrained=False)
        self.model.to(self.device)
        # バックボーンをImageEncoderと共有
        self.model.backbone = ResnetEncoder(self.device)
        self.model.head.classification_head.num_classes = num_classes+1

    def forward(self, image):
        return self.model(image)
    

