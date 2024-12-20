import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

class ReconstructionLoss(nn.Module):
    def __init__(self, device="cuda", original_image, reconstructed_image):
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
    def __init__(self, device="cuda", )




