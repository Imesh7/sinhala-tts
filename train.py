from copy import copy
import gc
import os
from pathlib import Path

import torch
import torch.nn as nn
import torchaudio
from sinlib import Tokenizer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from vocos import Vocos

from dataset.datamodule import DataModule
from inference import normalize_audio_for_save, process_audio, tokenize_text
from utils import uniquify
from zipvoice.utils.checkpoint import load_checkpoint, save_checkpoint
from zipvoice.utils.common import prepare_audio_input, sampling_time
from zipvoice.zipformer.scaling import ScheduledFloat
from zipvoice.zipvoice import ZipVoice

BATCH_SIZE = 4
ACCUMULATION_STEPS = 8
NUM_EPOCHS = 120


def update_batch_size(model: nn.Module, batch_size: int):
    if hasattr(model, "batch_size") and isinstance(model.batch_size, ScheduledFloat):
        model.batch_size.set_batch_size(batch_size)
    for child in model.children():
        update_batch_size(child, batch_size)


def train():
    n_mels = int(os.getenv("N_MELS", "100"))
    hop_length = int(os.getenv("HOP_LENGTH", "256"))
    n_fft = int(os.getenv("N_FFT", "1024"))
    sample_rate = int(os.getenv("SAMPLE_RATE", "24000"))

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
        n_mels=n_mels,
        hop_length=hop_length,
        n_fft=n_fft,
        sample_rate=sample_rate,
    )

    if len(train_dataloader) == 0:
        print(
            f"Warning: Training dataloader is empty. No audio files found in {file_path}."
        )
        return

    model = ZipVoice(feat_dim=n_mels, vocab_size=tokenizer.vocab_size).to(device)
    num_epochs = NUM_EPOCHS
    batch_size_idx = 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=1e-4,
        epochs=num_epochs,
        steps_per_epoch=max(1, len(train_dataloader) // ACCUMULATION_STEPS),
        pct_start=0.1,
    )

    writer = SummaryWriter(log_dir=log_dir)
    scaler = torch.amp.GradScaler() if device.type == "cuda" else None

    last_checkpoint = 500
    checkpoint_path = checkpoint_dir / f"checkpoint_step{last_checkpoint}.pth"
    model, optimizer, start_epoch = load_checkpoint(
        model, optimizer, str(checkpoint_path)
    )
    cfg_drop_ratio = 0.2

    model_avg = copy.deepcopy(model)
    model_avg.requires_grad_(False)

    for epoch in tqdm(range(start_epoch, num_epochs)):
        model.train()
        optimizer.zero_grad()
        epoch_losses = []

        for batch_idx, batch_data in enumerate(train_dataloader):
            mel_spec = batch_data["mel_specs"].to(device)
            text_tokens = batch_data["text_tokens"]
            batch_size = mel_spec.size(0)

            features, feature_lens = prepare_audio_input(mel_spec, device=device)
            t = sampling_time(batch_size, device=device, is_training=True)
            noise = torch.randn_like(features).to(device)

            loss = model(
                text_tokens,
                features,
                feature_lens,
                noise=noise,
                t=t,
                cfg_drop_ratio=cfg_drop_ratio,
                device=device,
            )

            update_batch_size(model, batch_size=batch_size_idx)

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            loss_val = loss.item()

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                update_ema(model, model_avg)

                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                batch_size_idx += 1

                epoch_losses.append(loss_val)
                writer.add_scalar("Loss/train", loss_val, batch_size_idx)
                writer.add_scalar("LR", scheduler.get_last_lr()[0], batch_size_idx)

                if batch_size_idx % 500 == 0:
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

            if batch_idx % 10 == 0:
                avg_loss = sum(epoch_losses[-10:]) / max(len(epoch_losses[-10:]), 1)
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] "
                    f"Batch [{batch_idx}/{len(train_dataloader)}] "
                    f"Loss: {loss_val:.4f} (avg: {avg_loss:.4f}) "
                    f"LR: {scheduler.get_last_lr()[0]:.2e}"
                )

        validation_output(
            model=model,
            epoch=epoch,
            val_dataloader=val_dataloader,
            writer=writer,
            device=device,
        )
        sampling_during_training(model_avg, tokenizer, vocos, device, checkpoint_dir)


# Expotential Moving Average (EMA) update for model parameters
def update_ema(model, model_avg, decay=0.999):
    with torch.no_grad():
        for p, p_avg in zip(model.parameters(), model_avg.parameters()):
            # p_avg.data.lerp_(p.data, 1-decay)  # p_avg = decay*p_avg + (1-decay)*p
            p_avg.data.mul_(decay).add_(p.data, alpha=1 - decay)


def validation_output(model, epoch, val_dataloader, writer, device: torch.device):
    model.eval()
    val_losses = []
    with torch.no_grad():
        for batch_data in val_dataloader:
            mel_spec = batch_data["mel_specs"].to(device)
            text_tokens = batch_data["text_tokens"]
            batch_size = mel_spec.size(0)

            features, feature_lens = prepare_audio_input(mel_spec, device=device)
            t = sampling_time(batch_size, device=device)

            noise = torch.randn_like(features).to(device)  # random noise

            val_loss = model(
                text_tokens,
                features,
                feature_lens,
                noise=noise,
                t=t,
                device=device,
            )

            val_losses.append(val_loss.item())

    avg_val_loss = sum(val_losses) / len(val_losses)
    writer.add_scalar("Loss/val", avg_val_loss, epoch)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    model.train()


def sampling_during_training(
    model: ZipVoice,
    tokenizer: Tokenizer,
    vocos: Vocos,
    device: torch.device,
    checkpoint_dir: Path,
):
    prompt_text = "ඒක නිසා මම"
    target_text = "ඔබට කොහොමද ඉන්නේ"
    prompt_audio = Path("/content/drive/My Drive/audio.wav")
    output_path = checkpoint_dir / "training_sample.wav"

    n_mels = int(os.getenv("N_MELS", "100"))
    hop_length = int(os.getenv("HOP_LENGTH", "256"))
    n_fft = int(os.getenv("N_FFT", "1024"))
    sample_rate = int(os.getenv("SAMPLE_RATE", "24000"))

    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = Tokenizer.from_pretrained("Ransaka/sinlib")

    prompt_text_tokens = tokenize_text(tokenizer, prompt_text)
    target_text_tokens = tokenize_text(tokenizer, target_text)

    model.to(device)

    feature_mel_spec = (
        process_audio(
            prompt_audio,
            n_mels=n_mels,
            hop_length=hop_length,
            n_fft=n_fft,
            sample_rate=sample_rate,
        )
        .unsqueeze(0)
        .to(device)
    )
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
        audio = vocos.decode(torch.exp(generated_mel)).cpu()
        peak = audio.abs().max().item()
        rms = audio.pow(2).mean().sqrt().item()
        print(f"Decoded audio stats: peak={peak:.6f}, rms={rms:.6f}")
        if peak < 1e-4:
            print(
                "Warning: decoded waveform is almost silent. This usually means the "
                "TTS model output and vocoder features do not match."
            )
        audio = normalize_audio_for_save(audio)

    output_path = Path(uniquify(str(output_path)))
    torchaudio.save(output_path, audio.squeeze(1), sample_rate)
    model.train()


if __name__ == "__main__":
    train()
