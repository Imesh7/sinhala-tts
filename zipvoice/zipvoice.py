from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from zipvoice.utils.common import condition_time_mask, pad_labels, pad_mask
from zipvoice.zipformer.zipformer import Zipformer
import numpy as np
import math


class ZipVoice(nn.Module):
    def __init__(
        self,
        down_sample_factors: List[int] = [1, 2, 4, 2, 1],
        vector_field_estimator_num_layers: List[int] = [2, 2, 4, 4, 4],
        feat_dim: int = 100,  # This should match the n_mels used in the TTSDataset
        text_emb_dim: int = 192,
        text_encoder_dim: int = 192,
        encoder_dim: int = 512,
        pos_dim: int = 48,
        q_head_dim: int = 32,
        v_head_dim: int = 12,
        vocab_size: int = 754,
        text_encoder_num_layers: int = 4,
        text_feed_forward_dim: int = 512,
        vfe_feed_forward_dim: int = 1536,
        time_emb_dim: int = 192,
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
            feed_forward_dim=vfe_feed_forward_dim,
            time_emb_dim=time_emb_dim,
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
        '''
        t: the time step, with the shape (batch, 1, 1).
        '''

        emb, token_lens = self.text_encode(
            tokens=tokens, device=device
        )  # [B, seq_len, text_emb_dim]

        (text_cond, pad_mask) = self.text_conditioning(
            text_emb=emb,
            features_lens=feature_lens,
            token_lens=token_lens,
            device=device,
        )

        condition_time_masked = condition_time_mask(
            features_lens=feature_lens,
            mask_percent=(0.7, 0.9),
            max_len=features.size(1),
        )

        speech_cond = torch.where(condition_time_masked.unsqueeze(-1), 0, features)

        # x_1 -> target
        # x_0 -> noise
        # x_t -> predicted noise
        x_t = t * features + (1 - t) * noise
        u_t = x_t - noise

        combined = torch.cat([x_t, text_cond, speech_cond], dim=2)
        
        while t.dim() > 1 and t.size(-1) == 1:
            t = t.squeeze(-1)
        # Handle t with a single value: expand to the size of batch size.
        if t.dim() == 0:
            t = t.repeat(x_t.shape[0])

        v_t = self.vector_field_estimator(
            x=combined, t=t, padding_mask=pad_mask, device=device
        )

        loss_mask = condition_time_masked & (~pad_mask)

        loss = torch.mean((v_t[loss_mask] - u_t[loss_mask]) ** 2)
        return loss

    def text_encode(
        self,
        tokens: List[List[int]],
        device: torch.device = None,
    ):
        """
        pad_id is '0' for the sinlib tokenizer
        """

        token_pad = pad_labels(tokens, pad_id=0, device=device)
        emb = self.emb(torch.tensor(token_pad, dtype=torch.int64).to(device))

        text_lengths = [len(token) for token in tokens]
        tokens_lens = torch.tensor(text_lengths, dtype=torch.int64, device=device)

        tokens_padding_mask = pad_mask(tokens_lens, emb.shape[1], device=device)

        x = self.text_encoder(
            x=emb, t=None, padding_mask=tokens_padding_mask, device=device
        )
        return x, tokens_lens

    def token_to_emb(self, tokens: List[List[int]], device: torch.device):
        x = self.emb(torch.tensor(tokens, dtype=torch.int64).to(device))
        return x

    def text_indexing(
        self, durations: List[List[int]], num_frames: int, device: torch.device
    ):
        durations = [x + [num_frames - sum(x)] for x in durations]
        batch_size = len(durations)

        ans = torch.zeros((batch_size, num_frames), dtype=torch.int64)

        for b in range(batch_size):
            cur_frame = 0
            for i, d in enumerate(durations[b]):
                ans[b, cur_frame : cur_frame + d] = i
                cur_frame += d
        return ans

    def text_conditioning(
        self,
        text_emb: torch.Tensor,
        features_lens: torch.Tensor,  # shape is [batch_size,]
        token_lens: torch.Tensor,
        device: torch.device,
    ):
        num_frames = int(features_lens.max())
        print(
            f"Num frames: {num_frames}, Text emb shape: {text_emb.shape}, Token lens shape: {token_lens.shape}, Features lens shape: {features_lens.shape}"
        )
        avg_upsampled_durations = self.average_upsample(
            token_lens=token_lens, features_lens=features_lens
        )
        token_indexing = self.text_indexing(
            durations=avg_upsampled_durations, num_frames=num_frames, device=device
        )

        pad_mask = pad_mask(features_lens, num_frames, device=device)

        text_condition = torch.gather(
            text_emb,
            dim=1,
            index=token_indexing.unsqueeze(-1).expand(
                text_emb.size(0), num_frames, text_emb.size(-1)
            ),
        )  # [batch, num_frames, text_emb_dim]

        return (text_condition, pad_mask)

    def average_upsample(self, token_lens: torch.Tensor, features_lens: torch.Tensor):
        res = []
        print(
            f"Token lens shape: {token_lens.shape}, Features lens shape: {features_lens.shape}"
        )
        for i in range(len(features_lens)):
            avg_token_duration = features_lens[i] // token_lens[i]
            res.append(
                [avg_token_duration] * token_lens[i]
            )  # if average 11 , it shows like this -> [11, 11, 11]
        return res


def speech_infilling_masking(feature_lens: torch.Tensor, mask_ratio, spans=3):
    target = feature_lens
    s_size = len(feature_lens)
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