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
        self.wav_dir = self.data_path / "audios"
        self.metadata_path = self.data_path / "dataset.csv"
        self.tokenizer = tokenizer

        # Load metadata (adjust format to your actual CSV)
        self.metadata = pd.read_csv(
            self.metadata_path,
            sep="|",
            header=None,
            names=["sentence", "audio"],
        )

        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=1.0,
            normalized=False,
            center=True,
        )
        # Resampler will be created on-demand
        self.resampler = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 1. Tokenize text
        text = row["sentence"]
        text_tokens = self.tokenizer(text).input_ids  # list of ints

        # 2. Load audio with librosa
        wav_filename = row["audio"]
        # if not wav_filename.endswith(".wav"):
        #     wav_filename += ".wav"

        # wav_path = self.wav_dir / wav_filename
        waveform_np, sr = librosa.load(wav_filename, sr=None)  # keep original sr

        # Convert to tensor and ensure shape [channels, time]
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() == 2:  # stereo: [time, channels]
            waveform = waveform.T  # -> [channels, time]
        else:  # mono: [time]
            waveform = waveform.unsqueeze(0)  # -> [1, time]

        # Resample if needed
        if sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self.resampler(waveform)

        # 3. Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)  # [1, n_mels, time]

        # Convert to log-mel (natural log)
        mel_log = mel_spec.clamp(min=1e-7).log()  # [1, n_mels, time]
        mel_log = mel_log.squeeze(0)  # [n_mels, time]

        return {
            "text_tokens": text_tokens,
            "mel_spec": mel_log,
            "text": text,
            "filename": wav_filename,
        }