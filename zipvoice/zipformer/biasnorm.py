import torch
import torch.nn as nn


class BiasNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super(BiasNorm, self).__init__()
        self.bias = nn.Parameter(torch.zeros(dim))
        self.r = nn.Parameter(torch.zeros(dim)) # scalar or weight
        self.eps = eps

    def forward(self, x):
        val = x - self.bias
        vari = val.pow(2).mean(dim=-1, keepdim=True)
        return x / (torch.sqrt(vari + self.eps)) * torch.exp(self.r)