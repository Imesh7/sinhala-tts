from pathlib import Path

import librosa
import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset

from utils import mel_spectrogram
import tqdm


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
        self.metadata_path = self.data_path / "dataset_clean.csv"
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate

        MAX_FRAMES_FILTER = 1200 # max audio size is 12 seconds

        # Load metadata (adjust format to your actual CSV)
        self.metadata = pd.read_csv(
            self.metadata_path,
            sep=",",
            header=0,
            names=["sentence", "audio"],
            engine='python',
        )
        
        # Path("/kaggle/working/length_cache.csv") or
        length_cache_path = Path("/teamspace/studios/this_studio/length_cache.csv")
        
        # This is compute frames of the audio files to remove long audios that consume more vram
        if length_cache_path.exists():
            lengths_df = pd.read_csv(length_cache_path)
            self.metadata["frames"] = lengths_df["frames"]
        else:
            print("⚠️ Computing lengths (one-time cost)...")

            frames = []
            for path in tqdm(self.metadata["audio"]):
                wav_path = self.wav_dir / path.replace("\\", "/")
                # print(wav_path)
                try:
                    print(wav_path)
                    y, _ = librosa.load(wav_path, sr=self.sample_rate)
                    frames.append(len(y) // hop_length)
                except:
                    print("librosa read error")
                    frames.append(0)

            self.metadata["frames"] = frames

            pd.DataFrame({"frames": frames}).to_csv(length_cache_path, index=False)
            print("✅ Saved length cache")

        # before = len(self.metadata)
        self.metadata = self.metadata[self.metadata["frames"] <= MAX_FRAMES_FILTER]

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
        
        # mean & std depends on audio data
        # so we have to compute them on our dataset
        # used `compute_mel_stats` function to compute them
        self.mel_mean = -1.7977
        self.mel_std = 2.0155

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 1. Tokenize text
        text = row["sentence"]
        text_tokens = self.tokenizer(text).input_ids

        wav_filename = row["audio"]
        wav_path = self.wav_dir / wav_filename.replace("\\", "/")
        wav, sr = librosa.load(str(wav_path), sr=None, mono=True)
        w = torch.from_numpy(wav).float()

        if w.dim() == 1:
            w = w.unsqueeze(0)
        elif w.size(0) > 1:
            w = w.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            if self.resampler is None or getattr(self.resampler, "orig_freq", None) != sr:
                self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            w = self.resampler(w)
        
        # Trim silence
        wav_np = w.squeeze(0).numpy()
        trimmed_np, _ = librosa.effects.trim(
            wav_np,
            top_db=30,
            frame_length=512,
            hop_length=128,
        )
        w = torch.from_numpy(trimmed_np).unsqueeze(0)

        w = w / (w.abs().max() + 1e-8)

        mel = self.mel_transform(w)
        mel = torch.clamp(mel, min=1e-7).log().squeeze(0)
        
        # Mel normalize
        mel = (mel - self.mel_mean) / self.mel_std

        return {
            "text_tokens": text_tokens,
            "mel_spec": mel,
            "text": text,
            "filename": wav_filename,
        }
