import torch
import torch.nn as nn
import timm
import torchvision

class ResnetEncoder(nn.Module):
    def __init__(self, model_name="resnet50",output_channels=256):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.output_channels = output_channels
        # 1x1畳み込み層でチャネル数を256に変換
        self.conv1x1 = nn.Conv2d(2048, self.output_channels, kernel_size=1)

    def forward(self, x):
        x = self.model(x)   
        x = self.conv1x1(x)
        return x
    
    


    