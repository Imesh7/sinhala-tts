import torch
import torch.nn as nn


# swooshR & swooshR avtivation fuinctions
class Swoosh(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    
    def forward(self, x):
        swoosh_r = torch.log(1 + torch.exp(x -1)) * (0.08 * x) *  0.313261687
        swoosh_l = torch.log(1 + torch.exp(x - 4)) * (0.08 * x) *  0.035
        return swoosh_r + swoosh_l