import torch
import torch.nn as nn

class Recostruction(nn.Module):
    def __init__(self, encoder_outchannels):
        super().__init__()
        self.encoder_outchannels = encoder_outchannels
        self.decoder = nn.Sequential(
            # upsampling and conv
            
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),


            nn.ConvTranspose2d(64, 3, kernel_size=3, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(x)
    
