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
        self.model.head.classification_head.cls_logits = nn.Conv2d(256, num_classes+1, kernel_size=3, stride=1, padding=1)

    def forward(self, image, target):
        return self.model(image, target)
    

