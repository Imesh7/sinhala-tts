from typing import List, Tuple

import torch


def prepare_audio_input(mel_spec, feat_scale=1.0, device=torch.device):
    """
    Prepare audio input for the model.
    mel_spec is audio mel spectrogram of shape [BATCH, NUM_FRAMES, TIME]
    tokens is text token ids of shape [B, N]
    """
    features = mel_spec.permute(0, 2, 1) * feat_scale # [BATCH, TIME, NUM_FRAMES]
    features_lens = torch.full((features.size(0),), features.size(2), dtype=torch.int64, device=device)

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
    y = [token_ids.tolist() + [pad_id] for token_ids in y]
    length = max([len(token_ids) for token_ids in y])
    y = [token_ids + [pad_id] * (length - len(token_ids)) for token_ids in y]
    return torch.tensor(y, dtype=torch.int64, device=device)


def condition_time_mask(
    features_lens: torch.Tensor,
    mask_percent: Tuple[float, float],
    max_len: int = 0,
) -> torch.Tensor:
    """
    Apply Time masking.
    Args:
        features_lens:
            input tensor of shape ``(B)``
        mask_size:
            the width size for masking.
        max_len:
            the maximum length of the mask.
    Returns:
        Return a 2-D bool tensor (B, T), where masked positions
        are filled with `True` and non-masked positions are
        filled with `False`.
    """
    mask_size = (
        torch.zeros_like(features_lens, dtype=torch.float32).uniform_(*mask_percent)
        * features_lens
    ).to(torch.int64)
    mask_starts = (
        torch.rand_like(mask_size, dtype=torch.float32) * (features_lens - mask_size)
    ).to(torch.int64)
    mask_ends = mask_starts + mask_size
    max_len = max(max_len, features_lens.max())
    seq_range = torch.arange(0, max_len, device=features_lens.device)
    mask = (seq_range[None, :] >= mask_starts[:, None]) & (
        seq_range[None, :] < mask_ends[:, None]
    )
    return mask

def pad_mask(lengths: torch.Tensor, max_length: int, device: torch.device):
    max_len = max(max_length, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)

    return expaned_lengths >= lengths.unsqueeze(-1)