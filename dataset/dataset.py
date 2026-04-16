from pathlib import Path

import librosa
import pandas as pd
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from numpy import np


class TTSDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        sample_rate: int = 24000,
        n_mels: int = 100,
        hop_length: int = 256,
        n_fft: int = 1024,
    ):
        self.data_path = Path(data_path)
        self.wav_dir = self.data_path / "wavs"
        self.metadata_path = self.data_path / "metadata.csv"
        self.tokenizer = tokenizer

        self.metadata = pd.read_csv(
            self.metadata_path,
            sep="|",
            header=None,
            names=["filename", "singlish_text", "sinhala_text", "others"],
            encoding="utf-8",
        )

        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
            normalized=True,
        )
        self.resampler = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        text = row["sinhala_text"]
        text_tokens = self.tokenizer(text).input_ids

        wav_filename = row["filename"]
        if not wav_filename.endswith(".wav"):
            wav_filename += ".wav"

        wav_path = self.wav_dir / wav_filename
        waveform_np, sr = librosa.load(wav_path, sr=None)

        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() == 2:
            waveform = waveform.T
        else:
            waveform = waveform.unsqueeze(0)

        if sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self.resampler(waveform)

        mel_spec = self.mel_transform(waveform)
        logmel = mel_spec.clamp(min=1e-7).log()
        mel_log = logmel.squeeze(0)

        return {
            "text_tokens": text_tokens,
            "mel_spec": mel_log,
            "text": text,
            "filename": wav_filename,
        }
