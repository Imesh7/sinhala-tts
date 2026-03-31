import os
from turtle import pd
from anyio import Path
import torch
from torch.utils.data import DataLoader, Dataset
import librosa
import torchaudio
import torchaudio.transforms as T


class TTSDataset(Dataset):
    def __init__(
        self,
        data_path,
        tokenizer,
        sample_rate=22050,
        n_mels=80,
        hop_length=256,
        n_fft=1024,
    ):
        self.data_path = Path(data_path)
        self.wav_dir = self.data_path / "wavs"
        self.metadata_path = self.data_path / "metadata.csv"
        self.tokenizer = tokenizer

        # Load metadata (adjust format to your actual CSV)
        self.metadata = pd.read_csv(
            self.metadata_path,
            sep="|",
            header=None,
            names=["filename", "singlish_text", "sinhala_text", "others"],
        )

        self.sample_rate = sample_rate
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            normalized=True,
        )
        # Resampler will be created on-demand
        self.resampler = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 1. Tokenize text
        text = row["sinhala_text"]
        text_tokens = self.tokenizer(text).input_ids  # list of ints

        # 2. Load audio with librosa
        wav_filename = row["filename"]
        if not wav_filename.endswith(".wav"):
            wav_filename += ".wav"

        wav_path = self.wav_dir / wav_filename
        waveform_np, sr = librosa.load(wav_path, sr=None)  # keep original sr

        # Convert to tensor and ensure shape [channels, time]
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() == 2:                # stereo: [time, channels]
            waveform = waveform.T              # -> [channels, time]
        else:                                  # mono: [time]
            waveform = waveform.unsqueeze(0)   # -> [1, time]

        # Resample if needed
        if sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = self.resampler(waveform)

        # 3. Compute mel-spectrogram
        mel_spec = self.mel_transform(waveform)          # [1, n_mels, time]
        # Convert to dB scale (dynamic range 0 to -80 dB)
        mel_spec = torchaudio.transforms.AmplitudeToDB(top_db=80.0)(mel_spec)
        mel_spec = mel_spec.squeeze(0)                   # [n_mels, time]

        # Normalize to [0,1] (assuming dB range -80..0)
        mel_spec = (mel_spec + 80) / 80

        return {
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "mel_spec": mel_spec,
            "text": text,
            "filename": wav_filename,
        }