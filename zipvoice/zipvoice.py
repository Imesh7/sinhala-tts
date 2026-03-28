from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from zipvoice.zipformer.zipformer import Zipformer
import numpy as np
import math


class ZipVoice(nn.Module):
    def __init__(
        self,
        down_sample_factors=[1, 2, 4, 2, 1],
        feat_dim: int = 100,
        text_emb_dim: int = 192,
        text_encoder_dim: int = 192,
        encoder_dim: int = 512,
        pos_dim: int = 48,
        q_head_dim: int = 32,
        v_head_dim: int = 12,
    ):
        super(ZipVoice, self).__init__()

        self.text_encoder = Zipformer(
            feat_dim,
            text_emb_dim,
            down_sample_factors,
            text_encoder_dim,
            pos_dim=pos_dim,
            q_head_dim=q_head_dim,
            v_head_dim=v_head_dim,
        )

        self.vector_field_estimator = Zipformer(
            feat_dim * 3,
            feat_dim,
            down_sample_factors,
            encoder_dim,
            pos_dim=pos_dim,
            q_head_dim=q_head_dim,
            v_head_dim=v_head_dim,
        )

    def forward(self, x: List[List[int]], features: torch.Tensor, noise, t: int):
        x = self.text_encoder(x)

        text_cond = average_upsample(x, features, fill_value=0)

        speech_cond = speech_infilling_masking(
            sound_emb=features, mask_ratio=0.5, spans=3
        )

        # x_1 -> target
        # x_0 -> noise
        # x_t -> predicted noise
        x_t = t * features + (1 - t) * noise
        u_t = x_t - noise

        combined = torch.cat([x_t, text_cond, speech_cond], dim=-1)

        v_t = self.vector_field_estimator(combined, x_t)

        loss = F.mse_loss(v_t, u_t)
        return x, loss


def average_upsample(text_emb, sound_emb, fill_value):
    n = text_emb.size(1)
    t = sound_emb.size(1)

    d = n // t
    upsampled_emb = text_emb.repeat_interleave(d, dim=1)

    if t > (d * n):
        upsampled_emb = F.pad(upsampled_emb, (0, 0, 0, t - d * n), value=fill_value)
    return upsampled_emb


def speech_infilling_masking(sound_emb, mask_ratio, spans=3):
    target = sound_emb.copy()
    s_size = len(sound_emb)
    mask_size = math.ceil(s_size * mask_ratio)
    mask = np.zeros(s_size)

    span_lengths = np.random.multinomial(mask_size, np.ones(spans) / spans, size=1)

    for span_len in span_lengths[0]:
        if span_len == 0:
            continue

        start = np.random.randint(0, s_size - span_len)
        mask[start : start + span_len] = 1

    masked_input = target * (1 - mask.unsqueeze(-1))

    return masked_input, mask, target
