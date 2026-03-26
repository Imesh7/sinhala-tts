from dataset.datamodule import DataModule
from zipvoice.zipvoice import ZipVoice
import torch
import torch.nn as nn

def train():

    dataloader = DataModule("path/to/data").dataloader(batch_size=32)  # Create a dataloader for the training data
    model = ZipVoice()  # Initialize the ZipVoice model 
    # Training loop for the ZipVoice model
    num_epochs = 10
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            noise = torch.randn_like(inputs)  # Add noise to the inputs
            outputs = model(inputs, noise)  # Forward pass with noise
            loss_val = loss(outputs, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()