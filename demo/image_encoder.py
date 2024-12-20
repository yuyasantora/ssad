import torch
import torch.nn as nn
import timm
import torchvision

class ResnetEncoder(nn.Module):
    def __init__(self, model_name="resnet50",output_channels=2048):
        super().__init__()
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-2])
        self.output_channels = output_channels

    def forward(self, x):
        return self.model(x)
    


    