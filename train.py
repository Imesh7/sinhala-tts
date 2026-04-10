import gc
from pathlib import Path

import torchaudio

from dataset.datamodule import DataModule
from inference import adapt_mel_for_vocos, infer_vocos_mel_dim, normalize_audio_for_save, normalized_db_to_vocos_features, process_audio, tokenize_text
from utils import uniquify
from zipvoice.utils.checkpoint import load_checkpoint, save_checkpoint
from zipvoice.utils.common import prepare_audio_input, sampling_time
from zipvoice.zipformer.scaling import ScheduledFloat
from zipvoice.zipvoice import ZipVoice
import torch
import torch.nn as nn
from sinlib import Tokenizer
from torch.utils.tensorboard import SummaryWriter
import tqdm
from vocos import Vocos

BATCH_SIZE = 32
SAMPLE_RATE = 24000
N_FFT = 1024
HOP_LENGTH = 256
TOP_DB = 80.0
VOCOS_SAMPLE_RATE = 24000
N_MELS = 100
VALIDATION_SET_PERCENTAGE = 0.1


def update_batch_size(model: nn.Module, batch_size: int):
    if hasattr(model, "batch_size") and isinstance(model.batch_size, ScheduledFloat):
        model.batch_size.set_batch_size(batch_size)
    for child in model.children():
        update_batch_size(child, batch_size)


def train():

    dataset_base_path = Path("/content/drive/MyDrive/nirvana_dataset")
    file_path = dataset_base_path

    checkpoint_dir = Path("/content/drive/MyDrive/sinhala-tts-checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = Path("/content/drive/MyDrive/sinhala-tts-logs")
    log_dir.mkdir(exist_ok=True)

    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
    vocos.eval()

    train_dataloader, val_dataloader = DataModule(file_path, tokenizer).dataloader(
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        sample_rate=SAMPLE_RATE,
    )  # Create a dataloader for the training data

    # If the dataloader is still empty, raise an error or handle it.
    if len(train_dataloader) == 0:
        print(
            f"Warning: Training dataloader is empty. No audio files found in {file_path}."
        )
        return  # Exit or handle as appropriate

    model = ZipVoice(feat_dim=N_MELS).to(device)  # Initialize the ZipVoice model
    # Training loop for the ZipVoice model
    num_epochs = 30
    batch_size_idx = 0

    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=num_epochs,
        steps_per_epoch=len(train_dataloader),
        pct_start=0.1,  # Warm up 10% of training
    )

    writer = SummaryWriter(log_dir=log_dir)  # logs
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    last_checkpoint = 500
    checkpoint_path = checkpoint_dir / f"checkpoint_step{last_checkpoint}.pth"

    # Fix: Pass the correct path to load_checkpoint
    model, optimizer, start_epoch = load_checkpoint(
        model, optimizer, str(checkpoint_path)
    )
    cfg_drop_ratio = 0.2

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        epoch_losses = []
        for batch_idx, batch_data in enumerate(train_dataloader):
            mel_spec = batch_data["mel_specs"].to(device)
            text_tokens = batch_data["text_tokens"]

            batch_size = mel_spec.size(0)

            features, feature_lens = prepare_audio_input(mel_spec, device=device)

            t = sampling_time(batch_size, device=device, is_training=True)

            noise = torch.randn_like(features).to(device)  # random noise

            loss = model(
                text_tokens, features, feature_lens, noise=noise, t=t,cfg_drop_ratio=cfg_drop_ratio, device=device
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
                    f"Batch [{batch_idx}/{len(train_dataloader)}] "
                    f"Loss: {loss_val:.4f} (avg: {avg_loss:.4f}) "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

            if batch_idx % 10 == 0:
                print(
                    f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss_val.item():.4f}"
                )

            if batch_size_idx % 500 == 0 and batch_size_idx > 0:
                validation_output(
                    model=model,
                    epoch=epoch,
                    val_dataloader=val_dataloader,
                    writer=writer,
                    device=device
                )
                checkpoint_file_path = (
                    checkpoint_dir / f"checkpoint_step{batch_size_idx}.pt"
                )
                save_checkpoint(
                    model,
                    optimizer,
                    epoch,
                    loss_val,
                    batch_idx,
                    batch_size_idx,
                    checkpoint_file_path,
                    scheduler,
                    scaler,
                )
                
        sampling_during_training(model, tokenizer, vocos, device)


def validation_output(model, epoch, val_dataloader, writer, device: torch.device):

    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(val_dataloader):
            mel_spec = batch_data["mel_specs"].to(device)
            text_tokens = batch_data["text_tokens"].to(device)

            batch_size = mel_spec.size(0)

            features, feature_lens = prepare_audio_input(mel_spec, device=device)

            t = sampling_time(batch_size, device=device, is_training=True)

            noise = torch.randn_like(features).to(device)  # random noise

            val_loss = model(
                text_tokens, features, feature_lens, noise=noise, t=t, device=device
            )

            val_losses.append(val_loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    model.train()
    
    
def sampling_during_training(model:ZipVoice, tokenizer:Tokenizer,vocos: Vocos, device:torch.device):
    
    prompt_text = "ඒක නිසා මම"
    target_text = "ඔබට කොහොමද ඉන්නේ"
    prompt_audio = Path("/content/drive/My Drive/audio.wav")
    
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

    prompt_text_tokens = tokenize_text(tokenizer, prompt_text)
    target_text_tokens = tokenize_text(tokenizer, target_text)

    model.to(device)
  
    feature_mel_spec = process_audio(prompt_audio, n_mels=N_MELS).to(device)
    prompt_features, prompt_feature_lens = prepare_audio_input(
        feature_mel_spec, device=device
    )

    with torch.no_grad():
        generated_mel, _, _, _ = model.sample(
            tokens=target_text_tokens,
            prompt_tokens=prompt_text_tokens,
            prompt_features=prompt_features,
            prompt_feature_lens=prompt_feature_lens,
            speed=1,
            device=device,
        )

        generated_mel = generated_mel.permute(0, 2, 1)
        vocos_features = normalized_db_to_vocos_features(generated_mel)
        vocos_mel_dim = infer_vocos_mel_dim(vocos)
        vocos_features = adapt_mel_for_vocos(vocos_features, vocos_mel_dim)
        audio = vocos.decode(vocos_features).cpu()
        peak = audio.abs().max().item()
        rms = audio.pow(2).mean().sqrt().item()
        print(f"Decoded audio stats: peak={peak:.6f}, rms={rms:.6f}")
        if peak < 1e-4:
            print(
                "Warning: decoded waveform is almost silent. This usually means the "
                "TTS model output and vocoder features do not match."
            )
        audio = normalize_audio_for_save(audio)

    # output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = uniquify(output_path)
    torchaudio.save(output_path, audio.squeeze(1), VOCOS_SAMPLE_RATE)
    model.train()

if __name__ == "__main__":
    train()