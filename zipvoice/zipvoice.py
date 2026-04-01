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
        down_sample_factors: List[int] = [1, 2, 4, 2, 1],
        vector_field_estimator_num_layers: List[int] = [2, 2, 4, 4, 4],
        feat_dim: int = 100,
        text_emb_dim: int = 192,
        text_encoder_dim: int = 192,
        encoder_dim: int = 512,
        pos_dim: int = 48,
        q_head_dim: int = 32,
        v_head_dim: int = 12,
        vocab_size: int = 754,
        text_encoder_num_layers: int = 4,
        text_feed_forward_dim: int = 512,
        vec_feed_forward_dim: int = 1536,
    ):
        super(ZipVoice, self).__init__()

        self.text_encoder = Zipformer(
            d_in=text_emb_dim,
            d_out=feat_dim,
            down_sample_factor=1,
            num_encoder_layers=text_encoder_num_layers,
            encoder_dim=text_encoder_dim,
            pos_dim=pos_dim,
            q_head_dim=q_head_dim,
            v_head_dim=v_head_dim,
            feed_forward_dim=text_feed_forward_dim,
        )

        self.vector_field_estimator = Zipformer(
            d_in=feat_dim * 3,
            d_out=feat_dim,
            down_sample_factor=down_sample_factors,
            num_encoder_layers=vector_field_estimator_num_layers,
            encoder_dim=encoder_dim,
            pos_dim=pos_dim,
            q_head_dim=q_head_dim,
            v_head_dim=v_head_dim,
            feed_forward_dim=vec_feed_forward_dim,
            
        )

        self.emb = nn.Embedding(vocab_size, text_emb_dim)

    def forward(
        self,
        tokens: List[List[int]],
        features: torch.Tensor,
        feature_lens: torch.Tensor,
        noise: torch.Tensor,
        t: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:

        emb = self.text_encode(tokens=tokens, t=t, device=device)

        text_cond = self.text_conditioning(
            text_emb=emb,
            features_lens=feature_lens,
            device=device,
        )

        speech_cond = speech_infilling_masking(
            sound_emb=features, mask_ratio=0.5, spans=3
        )

        # x_1 -> target
        # x_0 -> noise
        # x_t -> predicted noise
        x_t = t * features + (1 - t) * noise
        u_t = x_t - noise

        combined = torch.cat([x_t, text_cond, speech_cond], dim=-1)

        v_t = self.vector_field_estimator(x=combined, t=x_t, device=device)

        loss = F.mse_loss(v_t, u_t)
        return loss

    def text_encode(
        self,
        tokens: List[List[int]],
        t: torch.Tensor = None,
        device: torch.device= None,
    ):
        x = self.emb(torch.tensor(tokens, dtype=torch.int64).to(device))
        x = self.text_encoder(x=x, t=t, device=device)
        return x

    def text_indexing(
        self, durations: List[List[int]], num_frames: int, device: torch.device
    ):
        durations = [x + [num_frames - sum(x)] for x in durations]
        batch_size = len(durations)

        ans = torch.zeros((batch_size, num_frames), dtype=torch.int64).to(device)

        for b in range(batch_size):
            cur_frame = 0
            for i, d in durations[b]:
                ans[b, cur_frame : cur_frame + d] = i
                cur_frame += d
        return ans

    def text_conditioning(
        self,
        text_emb: torch.Tensor,
        features_lens: torch.Tensor,  # shape is [batch_size,]
        device: torch.device,
    ):
        num_frames = int(features_lens.max())
        avg_upsampled_durations = self.average_upsample(text_emb, features_lens)
        text_indexing = self.text_indexing(
            durations=avg_upsampled_durations, num_frames=num_frames, device=device
        )

        return torch.gather(
            text_emb,
            dim=1,
            index=text_indexing.unsqueeze(-1).expand(
                text_emb.size(0), num_frames, text_emb.size(-1)
            ),
        )  # [batch, num_frames, text_emb_dim]

    def average_upsample(self, text_emb: torch.Tensor, features_lens: torch.Tensor):
        res = []
        for i in range(len(features_lens)):
            token_len = len(text_emb[i])
            avg_token_duration = features_lens[i] // token_len
            res.append(
                [avg_token_duration] * token_len
            )  # if average 11 , it shows like this -> [11, 11, 11]
        return res


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
