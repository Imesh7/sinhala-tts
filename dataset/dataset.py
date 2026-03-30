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
        """
        Args:
            data_path: Path to nirvana_dataset folder (contains wav/ and metadata.csv)
            tokenizer: Your Sinhala tokenizer (e.g., from sinlib)
            sample_rate: Target sample rate (Nirvana is 22.05kHz, Vocos uses 24kHz)
            n_mels: Number of mel bands
        """
        self.data_path = Path(data_path)
        self.wav_dir = self.data_path / "wav"
        self.metadata_path = self.data_path / "metadata.csv"
        self.tokenizer = tokenizer

        # Load metadata CSV
        # Nirvana format: filename|transcription (e.g., "sinhala_0001|ආයුබෝවන්")
        self.metadata = pd.read_csv(
            self.metadata_path,
            sep="|",
            header=None,
            names=["filename", "singlish_text", "sinhala_text"],
        )

        # Audio transforms
        self.sample_rate = sample_rate
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
            normalized=True,
        )
        self.amplitude_db = T.AmplitudeToDB(st_ref=1.0, top_db=80.0)

        # Resampler (if needed - Nirvana is 22050, but you might want 24000 for Vocos)
        self.resampler = None

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]

        # 1. Get text and tokenize
        text = row["sinhala_text"]
        text_tokens = self.tokenizer(text).input_ids  # List of integers

        # 2. Load audio
        wav_filename = row["filename"]
        if not wav_filename.endswith(".wav"):
            wav_filename += ".wav"

        wav_path = self.wav_dir / wav_filename
        waveform, sr = torchaudio.load(wav_path)  # Returns [channels, time]

        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Resample if needed (Nirvana is 22050, Vocos expects 24000)
        if sr != self.sample_rate:
            if self.resampler is None or self.resampler.orig_freq != sr:
                self.resampler = T.Resample(sr, self.sample_rate)
            waveform = self.resampler(waveform)

        # 3. Convert to Mel-spectrogram
        mel_spec = self.mel_transform(waveform)  # [1, n_mels, time]
        mel_spec = self.amplitude_db(mel_spec)  # Convert to dB scale
        mel_spec = mel_spec.squeeze(0)  # [n_mels, time]

        # Normalize mel (optional but recommended)
        mel_spec = (mel_spec + 80) / 80  # Normalize to [0, 1] range

        return {
            "text_tokens": torch.tensor(text_tokens, dtype=torch.long),
            "mel_spec": mel_spec,
            "text": text,  # Keep for debugging
            "filename": wav_filename,
        }
