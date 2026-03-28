from dataset.datamodule import DataModule
from zipvoice.zipvoice import ZipVoice
import torch
import torch.nn as nn
from vocos import Vocos
from sinlib import Tokenizer

def train():
    
    dataloader = DataModule("path/to/data").dataloader(batch_size=32)  # Create a dataloader for the training data
    
    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")
    
    model = ZipVoice()  # Initialize the ZipVoice model 
    # Training loop for the ZipVoice model
    num_epochs = 10
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size_idx = 0
    
    # vocoder
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    for epoch in range(num_epochs):
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            noise = torch.randn_like(inputs)  # Add noise to the inputs
            outputs = model(inputs, noise)  # Forward pass with noise
            loss_val = loss(outputs, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            update_batch_size(model, batch_size=batch_size_idx)  # Update batch size for ScheduledFloat
            batch_size_idx += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss_val.item():.4f}")
            
if __name__ == "__main__":
    train()
    
    
def update_batch_size(model : nn.Module, batch_size: int):
    if hasattr(model, "batch_size"):
        model.batch_size = batch_size
    for child in model.children():
        update_batch_size(child, batch_size)