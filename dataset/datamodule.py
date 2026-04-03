import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from dataset.dataset import TTSDataset


class DataModule:
    def __init__(self, data_path, tokenizer):
        self.data_path = data_path
        self.tokenizer = tokenizer

    def dataloader(self, batch_size=32, shuffle=True, num_workers=2):
        dataset = TTSDataset(self.data_path, self.tokenizer)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=tts_collate_fn,  # Important for variable lengths
            pin_memory=True if torch.cuda.is_available() else False,
        )
    


# Collate function to handle variable length sequences
def tts_collate_fn(batch):
    """
    Pads text tokens and mel spectrograms to max length in batch
    """
    text_lengths = [len(item["text_tokens"]) for item in batch]
    mel_lengths = [item["mel_spec"].shape[-1] for item in batch]

    max_mel_len = max(mel_lengths)
    n_mels = batch[0]["mel_spec"].shape[0]


    # Pad mel specs
    mel_padded = torch.zeros(len(batch), n_mels, max_mel_len)
    for i, item in enumerate(batch):
        mel_padded[i, :, : item["mel_spec"].shape[-1]] = item["mel_spec"]

    return {
        "text_tokens": [item["text_tokens"] for item in batch],
        "mel_specs": mel_padded,
        "text_lengths": torch.tensor(text_lengths),
        "mel_lengths": torch.tensor(mel_lengths),
        "texts": [item["text"] for item in batch],
    }

