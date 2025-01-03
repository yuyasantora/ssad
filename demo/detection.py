import torch
import torch.nn as nn
from torchvision.models.detection import fcos_resnet50_fpn
from torchvision.models.detection.fcos import FCOSClassificationHead

from demo.image_encoder import ResnetEncoder

class FCOSDetector(nn.Module):
    def __init__(self, device="cuda",num_classes=32):
        super().__init__()
        self.device = device
        self.model = fcos_resnet50_fpn(pretrained=False)
        self.model.to(self.device)
        # バックボーンをImageEncoderと共有
        self.model.backbone = ResnetEncoder(self.device)
        # 分類ヘッドをクラス数+1に変更
        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head.classification_head.cls_logits = FCOSClassificationHead(in_channels=256, num_classes=num_classes+1, num_anchors=num_anchors)

    def forward(self, image, target):
        return self.model(image, target)
    

