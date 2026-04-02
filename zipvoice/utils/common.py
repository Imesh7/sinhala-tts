from typing import List

import torch


def prepare_audio_input(features, feat_scale=1.0, device=torch.device):
    """
    Prepare audio input for the model.
    features is audio mel spectrogram of shape [T, D]
    tokens is text token ids of shape [B, N]
    """
    features = features.unsqueeze(0) * feat_scale
    features_lens = torch.tensor([features.size(1)], device=device)

    return features, features_lens


def sampling_time(batch_size, device, is_training=False):
    if is_training:
        t = torch.rand(batch_size, 1, 1, device=device)
    else:
        t = (
            (torch.arange(batch_size, device=device) / batch_size)
            .unsqueeze(1)
            .unsqueeze(2)
        )
    return t


def to_tuple(x, downsampling_factor):
    if isinstance(x, int):
        x = (x,)
    if len(x) == 1:
        x = x * len(downsampling_factor)
    return x

def pad_labels(y: List[List[int]], pad_id: int, device: torch.device):
    """
    Pad the transcripts to the same length with zeros.

    Args:
      y: the transcripts, which is a list of a list

    Returns:
      Return a Tensor of padded transcripts.
    """
    y = [token_ids + [pad_id] for token_ids in y]
    length = max([len(token_ids) for token_ids in y])
    y = [token_ids + [pad_id] * (length - len(token_ids)) for token_ids in y]
    return torch.tensor(y, dtype=torch.int64, device=device)