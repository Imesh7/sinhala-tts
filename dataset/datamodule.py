import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset.dataset import TTSDataset


class DataModule(torch.utils):
    def __init__(self, data):
        self.data = data

    def dataloader(self, batch_size, shuffle=True):
        dataset = TTSDataset(self.data)

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader
    
