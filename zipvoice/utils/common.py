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