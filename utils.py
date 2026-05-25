import os
from pathlib import Path
import librosa
import torch
import torchaudio.transforms as T

from torch.utils.data import DataLoader
from tqdm import tqdm


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


def mel_spectrogram(
    n_mels: int, hop_length: int, n_fft: int, sample_rate: int
) -> T.MelSpectrogram:
    return T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=1.0,
        normalized=False,
        center=True,
    )


# Process Audio before Inference & training
def process_audio(
    file_path: Path,
    n_mels: int,
    hop_length: int,
    n_fft: int,
    sample_rate: int,
    mel_mean: float = -1.7977,
    mel_std: float = 2.0155,
) -> torch.Tensor:
    
    waveform_np, original_sample_rate = librosa.load(file_path, sr=None)
    waveform = torch.from_numpy(waveform_np).float()

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if original_sample_rate != sample_rate:
        resampler = T.Resample(orig_freq=original_sample_rate, new_freq=sample_rate)
        waveform = resampler(waveform)

    wav_np = waveform.squeeze(0).numpy()
    trimmed_np, _ = librosa.effects.trim(
        wav_np,
        top_db=30,
        frame_length=512,
        hop_length=128,
    )
    w = torch.from_numpy(trimmed_np).unsqueeze(0)

    w = w / (w.abs().max() + 1e-8)

    mel_transform = mel_spectrogram(
        n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, sample_rate=sample_rate
    )

    mel_spec = mel_transform(w)  # [1, n_mels, time]

    mel_log = torch.clamp(mel_spec, min=1e-7).log().squeeze(0)  # [1, n_mels, time]

    mel = (mel_log - mel_mean) / mel_std
    return mel


def compute_mel_stats(dataloader: DataLoader):

    running_sum = 0.0
    running_sq = 0.0
    count = 0

    for batch in tqdm(dataloader, desc="Computing mel stats"):
        mel = batch["mel_specs"]  # [n_mels, T]
        running_sum += mel.sum().item()
        running_sq += (mel**2).sum().item()
        count += mel.numel()

    mean = running_sum / count
    std = (running_sq / count - mean**2) ** 0.5
    print(f"mel_mean = {mean:.4f},  mel_std = {std:.4f}")
    return mean, std
