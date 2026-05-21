from pathlib import Path

import librosa
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from utils import mel_spectrogram


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
        self.root = Path(data_path)
        self.meta = pd.read_csv(
            self.root / "dataset.csv",
            sep=",",
            header=0,
            names=["sentence", "audio"],
            engine="python",
        )
        self.tok = tokenizer
        self.sample_rate = sample_rate
        self.mel = mel_spectrogram(
            n_mels=n_mels, hop_length=hop_length, n_fft=n_fft, sample_rate=sample_rate
        )
        self.rs = None

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        row = self.meta.iloc[idx]
        text_tokens = self.tok(row["sentence"]).input_ids
        wav_path = self.root / row["audio"].replace("\\", "/")
        wav, sr = librosa.load(str(wav_path), sr=None)
        w = torch.from_numpy(wav).float()

        if w.dim() == 1:
            w = w.unsqueeze(0)
        elif w.size(0) > 1:
            w = w.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            if self.rs is None or getattr(self.rs, "orig_freq", None) != sr:
                self.rs = torchaudio.transforms.Resample(sr, self.sample_rate)
            w = self.rs(w)

        mel = self.mel(w).clamp(min=1e-7).log().squeeze(0)

        return {"text_tokens": text_tokens, "mel_spec": mel, "text": row["sentence"]}
