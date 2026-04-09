import os
import torch


def load_checkpoint(model, optimizer, filename="checkpoint_step.pth"):
    if os.path.isfile(filename):
        print(f"Loading checkpoint '{filename}'")
        checkpoint = torch.load(filename)
        start_epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Resumed from epoch {start_epoch}")

        if optimizer is None:
            return model, None, start_epoch
        else:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            return model, optimizer, start_epoch
    else:
        print(f"No checkpoint found at '{filename}'")
        if optimizer is None:
            return model, None, 0
        return model, optimizer, 0


def save_checkpoint(
    model,
    optimizer,
    epoch,
    loss,
    batch_idx,
    batch_size_idx,
    checkpoint_file_path,
    scheduler,
    scaler=None,
):
    checkpoint = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "batch_count": batch_size_idx,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "loss": loss,
        "scaler_state_dict": scaler.state_dict() if scaler else None,
    }
    torch.save(
        checkpoint,
        checkpoint_file_path,
    )
    print(f"💾 Saved checkpoint: {checkpoint_file_path}")
