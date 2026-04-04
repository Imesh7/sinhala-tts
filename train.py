from pathlib import Path

from dataset.datamodule import DataModule
from zipvoice.utils.common import prepare_audio_input, sampling_time
from zipvoice.zipformer.scaling import ScheduledFloat
from zipvoice.zipvoice import ZipVoice
import torch
import torch.nn as nn
from vocos import Vocos
from sinlib import Tokenizer
import os
from torch.utils.tensorboard import SummaryWriter
import tqdm


def update_batch_size(model: nn.Module, batch_size: int):
    if hasattr(model, "batch_size") and isinstance(model.batch_size, ScheduledFloat):
        model.batch_size.set_batch_size(batch_size)
    for child in model.children():
        update_batch_size(child, batch_size)


def load_checkpoint(model, optimizer, filename="checkpoint_step.pth"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Resumed from epoch {start_epoch}")
        return model, optimizer, start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        return model, optimizer, 0


def train():

    dataset_base_path = Path("/content/drive/MyDrive/nirvana_dataset")
    file_path = dataset_base_path

    checkpoint_dir = Path("/content/drive/MyDrive/sinhala-tts-checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = Path("/content/drive/MyDrive/sinhala-tts-logs")
    log_dir.mkdir(exist_ok=True)

    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = DataModule(file_path, tokenizer).dataloader(
        batch_size=32
    )  # Create a dataloader for the training data

    # If the dataloader is still empty, raise an error or handle it.
    if len(dataloader) == 0:
        print(f"Warning: Dataloader is empty. No audio files found in {file_path}.")
        return  # Exit or handle as appropriate

    model = ZipVoice().to(device)  # Initialize the ZipVoice model
    # Training loop for the ZipVoice model
    num_epochs = 10
    batch_size_idx = 0

    # vocoder
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    vocos.eval()

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.1,  # Warm up 10% of training
    )

    writer = SummaryWriter(log_dir=log_dir)  # logs
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    last_checkpoint = 500
    checkpoint_path = checkpoint_dir / f"checkpoint_step{last_checkpoint}.pth"

    # Fix: Pass the correct path to load_checkpoint
    model, optimizer, start_epoch = load_checkpoint(
        model, optimizer, str(checkpoint_path)
    )

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        epoch_losses = []
        for batch_idx, batch_data in enumerate(dataloader):
            mel_spec = batch_data["mel_specs"].to(device)
            text_tokens = batch_data["text_tokens"].to(device)

            batch_size = mel_spec.size(0)

            features, feature_lens = prepare_audio_input(mel_spec, device=device)

            t = sampling_time(batch_size, device=device, is_training=True)

            noise = torch.randn_like(features).to(device)  # random noise

            loss = model(
                text_tokens, features, feature_lens, noise=noise, t=t, device=device
            )  # Forward pass with noise

            update_batch_size(
                model, batch_size=batch_size_idx
            )  # Update batch size for ScheduledFloat
            batch_size_idx += 1

            optimizer.zero_grad(set_to_none=True)

            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            loss_val = loss.item()
            epoch_losses.append(loss_val)
            writer.add_scalar("Loss/train", loss_val, batch_size_idx)
            writer.add_scalar("LR", scheduler.get_last_lr()[0], batch_size_idx)

            # Print every 10 batches
            if batch_idx % 10 == 0:
                avg_loss = sum(epoch_losses[-10:]) / len(epoch_losses[-10:])
                print(
                    f"Epoch [{epoch+1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(dataloader)}] "
                    f"Loss: {loss_val:.4f} (avg: {avg_loss:.4f}) "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss_val.item():.4f}"
                )

            if batch_size_idx % 500 == 0 and batch_size_idx > 0:
                ckpt_path = checkpoint_dir / f"checkpoint_step{batch_size_idx}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "batch_count": batch_size_idx,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": loss,
                        "scaler_state_dict": scaler.state_dict() if scaler else None,
                    },
                    ckpt_path,
                )
                print(f"💾 Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    train()
