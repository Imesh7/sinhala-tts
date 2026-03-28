from pathlib import Path

from dataset.datamodule import DataModule
from zipvoice.zipvoice import ZipVoice
import torch
import torch.nn as nn
from vocos import Vocos
from sinlib import Tokenizer
import os
from torch.utils.tensorboard import SummaryWriter

def train():

    file_path = '/content/drive/MyDrive/nirvana_dataset/'
    checkpoint_dir = Path("/content/drive/MyDrive/sinhala-tts-checkpoints/")
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = Path("/content/drive/MyDrive/sinhala-tts-logs")
    log_dir.mkdir(exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataModule(file_path).dataloader(batch_size=32)  # Create a dataloader for the training data
    
    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")
    
    model = ZipVoice().to(device)  # Initialize the ZipVoice model 
    # Training loop for the ZipVoice model
    num_epochs = 10
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    batch_size_idx = 0
    
    # vocoder
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocos.eval()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=1e-4, 
        epochs=10, 
        steps_per_epoch=len(dataloader),
        pct_start=0.1  # Warm up 10% of training
    )

    writer = SummaryWriter(log_dir=log_dir) # logs
    scaler = torch.amp.GradScaler() if device.type == 'cuda' else None

    last_checkpoint = 500
    checkpoint_path = checkpoint_dir / f"checkpoint_step{last_checkpoint}.pth"

    model , optimizer, start_epoch = load_checkpoint(model, optimizer,checkpoint_dir + "")

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_losses = []
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            noise = torch.randn_like(inputs).to(device)  # Add noise to the inputs
            outputs = model(inputs, noise)  # Forward pass with noise
            loss_val = loss(outputs, targets)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()
            update_batch_size(model, batch_size=batch_size_idx)  # Update batch size for ScheduledFloat
            batch_size_idx += 1

            # Backward
            optimizer.zero_grad(set_to_none=True)  # More memory efficient
            
            if scaler:
                scaler.scale(loss).backward()
                # Gradient clipping (prevents explosion)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            # Track loss
            loss_val = loss.item()
            epoch_losses.append(loss_val)
            writer.add_scalar("Loss/train", loss_val, batch_size_idx)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], batch_size_idx)
            
            # Print every 10 batches
            if batch_idx % 10 == 0:
                avg_loss = sum(epoch_losses[-10:]) / len(epoch_losses[-10:])
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Batch [{batch_idx}/{len(dataloader)}] "
                      f"Loss: {loss_val:.4f} (avg: {avg_loss:.4f}) "
                      f"LR: {scheduler.get_last_lr()[0]:.2e}")
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss_val.item():.4f}")

            if batch_size_idx % 500 == 0 and batch_size_idx > 0:
                ckpt_path = checkpoint_dir / f"checkpoint_step{batch_size_idx}.pt"
                torch.save({
                    'epoch': epoch,
                    'batch_idx': batch_idx,
                    'batch_count': batch_size_idx,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_val,
                    'scaler_state_dict': scaler.state_dict() if scaler else None,
                }, ckpt_path)
                print(f"💾 Saved checkpoint: {ckpt_path}")
            
if __name__ == "__main__":
    train()
    
    
def update_batch_size(model : nn.Module, batch_size: int):
    if hasattr(model, "batch_size"):
        model.batch_size = batch_size
    for child in model.children():
        update_batch_size(child, batch_size)



def load_checkpoint(model, optimizer, filename='checkpoint_step.pth.tar'):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Resumed from epoch {start_epoch}")
        return model, optimizer, start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return model, optimizer, 0