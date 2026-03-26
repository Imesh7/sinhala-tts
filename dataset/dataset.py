import os
import torch
from torch.utils.data import Dataset
import librosa


class TTSDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        # Load and preprocess the dataset here

        self.files = os.listdir(self.data_path)

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.files)

    def __getitem__(self, idx):
        # Return a single sample (text, features, noise) based on the index
        audio, _ = librosa.load(self.files[idx], sr=16000)  # Load audio file
        # Load and return the sample (example placeholder)
        transcribe = self.files[idx].split(".")[0]  # Assuming the filename is the transcription
        return audio, transcribe