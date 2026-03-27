from shutil import copy
from typing import List

import torch
import torch.nn as nn
import random
from torch import Tensor

from zipvoice.zipformer.biasnorm import BiasNorm
from zipvoice.zipformer.scaling import FloatLike, ScheduledFloat
from zipvoice.zipformer.swosh_activation import Swoosh


class Zipformer(nn.Module):
    def __init__(
        self,
        d_in,
        d_out,
        down_sample_factor: List[int] = [1, 2, 4, 2, 1],
        encoder_dim: int = 384,
        num_heads: int = 4,
        pos_dim: int = 48,
        q_head_dim: int = 32,
        v_head_dim: int = 12,
    ):
        super(Zipformer, self).__init__()
        self.conv_emb = ConvolutionalEmbedding(
            in_channels=d_in, out_channels=encoder_dim
        )

        self.in_proj = nn.Linear(d_in, encoder_dim)
        self.out_proj = nn.Linear(encoder_dim, d_out)

        encoder_layer = []
        num_encoder_layers = len(down_sample_factor)

        for i in range(num_encoder_layers):
            zipformer_block = ZipformerBlock(
                emb_dim=encoder_dim,
                pos_emb=pos_dim,
                num_heads=num_heads,
                q_head_dim=q_head_dim,
                v_head_dim=v_head_dim
            )
            encoder = ZipformerEncoder(zipformer_block)

            if down_sample_factor[i] > 1:
                downsampled_encoder = DownsampledZipformerEncoder(encoder_layer=encoder)

            encoder_layer.append(downsampled_encoder)

        self.encoder_layers = nn.ModuleList(encoder_layer)

    def forward(self, x, atten_masks=None):
        x = x.permute(1, 0, 2)
        x = self.in_proj(x)

        for layer in self.encoder_layers:
            x = layer(x, atten_masks)

        x = self.out_proj(x)
        x = x.permute(1, 0, 2)
        return x


class ZipformerEncoder(nn.Module):
    def __init__(self, encoder_layers):
        super().__init__()
        self.encoder_layer = encoder_layers

    def forward(self, x):
        _x = x
        for layer in self.encoder_layer:
            x = layer(x)
        return x + _x


class DownsampledZipformerEncoder(nn.Module):
    def __init__(self, encoder_layer, d_in):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.downsample = Downsample(downsample_factor=2)
        self.upsample = Upsample(upsample_factor=2)
        self.bypass = ByPass(emb_dim=d_in, skip_rate=0.1, straight_through_rate=0.1)

    def forward(self, x, atten_masks=None):
        x = self.downsample(x)
        x = self.encoder_layer(x, atten_masks)
        x = self.upsample(x)

        return self.bypass(x)


class Downsample(nn.Module):
    def __init__(self, downsample_factor):
        super().__init__()
        self.downsample = downsample_factor
        self.bias = nn.Parameter(torch.zeros(self.downsample))

    def forward(self, src: Tensor):
        (seq_len, batch, in_channels) = src.shape
        downsample = self.downsample
        downsampled_seq_len = (seq_len + downsample - 1) // downsample

        # Padding
        pad = downsampled_seq_len * downsample - seq_len
        src_extra = src[src.shape[0] - 1 :].expand(pad, src.shape[1], src.shape[2])
        src = torch.cat((src, src_extra), dim=0)
        assert src.shape[0] == downsampled_seq_len * downsample

        src = src.reshape(downsampled_seq_len, downsample, batch, in_channels)
        weights = self.bias.softmax(dim=0)
        weights = weights.unsqueeze(-1).unsqueeze(-1)
        downsampled_src = (src * weights).sum(dim=1)
        return downsampled_src


class Upsample(nn.Module):
    def __init__(self, upsample_factor):
        super().__init__()
        self.upsample = upsample_factor

    def forward(self, src: Tensor):
        (seq_len, batch_size, num_channels) = src.shape
        upsampled_seq_len = seq_len * self.upsample

        src = src.unsqueeze(1).expand(seq_len, self.upsample, batch_size, num_channels)
        src = src.reshape(upsampled_seq_len, batch_size, num_channels)
        return src


class ConvolutionalEmbedding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.conv_emb = nn.Sequential(
            conv_2d(
                in_channels=in_channels,
                out_channels=8,
                kernel_size=kernel_size,
                padding=padding,
                stride=(1, 2),
            ),
            nn.ReLU(),
            conv_2d(
                in_channels=8,
                out_channels=32,
                kernel_size=kernel_size,
                padding=padding,
                stride=(2, 2),
            ),
            nn.ReLU(),
            conv_2d(
                in_channels=32,
                out_channels=128,
                kernel_size=kernel_size,
                padding=padding,
                stride=(1, 2),
            ),
        )

        self.out_proj = nn.Linear(128, out_channels)
        self.bias_norm = BiasNorm(out_channels)

    def forward(self, x):
        # x -> (batch, time, channels)
        x = x.permute(0, 2, 1)  # (batch, channels, time)
        x = self.conv_emb(x)
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        x = self.out_proj(x)

        x = self.bias_norm(x)
        return x


def conv_2d(in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
    )


class ZipformerBlock(nn.Module):
    def __init__(self, emb_dim, pos_emb, num_heads, q_head_dim, v_head_dim):
        super().__init__()
        self.feed_forward_1 = ZipformerFeedForward()
        self.non_linear_attention = NonLinearAttention(channels=emb_dim, hidden_channels=(3 * emb_dim) // 4)
        self.self_attention_1 = RelativePositionalMultiHeadAttention(
            d_in=emb_dim,
            d_v_out=pos_emb,
            num_heads=num_heads,
            q_head_dim=q_head_dim,
        )

        self.convolution_1 = Convolution(channels=emb_dim, kernel_size=3)

        self.feed_forward_2 = ZipformerFeedForward()
        self.bypass_1 = ByPass(emb_dim=emb_dim, skip_rate=0.1, straight_through_rate=0.1)
        self.self_attention_2 = SelfAttention(emb_dim=emb_dim, d_v_out=v_head_dim)

        self.feed_forward_3 = ZipformerFeedForward()
        self.bias_norm = BiasNorm(emb_dim)
        self.bypass_2 = ByPass(emb_dim=emb_dim, skip_rate=0.1, straight_through_rate=0.1)

        self.convolution_2 = Convolution(channels=emb_dim, kernel_size=3)

    def forward(self, x, atten_masks=None):
        atten_weights = self.self_attention_1(x)

        x = x + self.feed_forward_1(x)
        x = x + self.non_linear_attention(x, atten_weights)
        x = x + self.self_attention_1(x, atten_weights)

        x = x + self.convolution_1(x)

        x = x + self.feed_forward_2(x)
        x = self.bypass_1(x, c)
        x = x + self.self_attention_2(x)

        x = x + self.convolution_2(x)

        x = x + self.feed_forward_3(x)

        x = self.bias_norm(x)
        x = self.bypass_2(x, c)


class ZipformerFeedForward(nn.Module):
    def __init__(self, dims=[512, 2048, 768], dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer_norm = nn.ModuleList()
        self.dropout = nn.Dropout(dropout)

        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i + 1]))

            if i < len(dims) - 2:
                self.layer_norm.append(nn.LayerNorm(dims[i + 1]))

    def forward(self, x):
        x = self.layers[0](x)
        x = self.layer_norm[0](residual)

        for i in range(1, len(self.layers) - 1):
            residual = x
            x = self.layers[i](x)
            x = self.layer_norm[i](x)
            x = self.dropout(x)
            x = x + residual

        x = self.layers[-1](x)
        return x


class NonLinearAttention(nn.Module):
    def __init__(self, channels, hidden_channels):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.in_proj = nn.Linear(channels, hidden_channels * 3, bias=True)
        self.tanh = nn.Tanh()
        self.out_proj = nn.Linear(hidden_channels, channels, bias=True)

    def forward(self, x : Tensor, atten_weights : Tensor=None):
        x = self.in_proj(x)
        (seq_len, batch_size, _) =  x.shape
        
        s, x, y = x.chunk(3, dim=2)
        s = self.tanh(s)
        
        s = s.unsqueeze(-1).reshape(seq_len, batch_size, self.hidden_channels)
        x = x * s
        
        (seq_len, batch_size, embed_dim) = x.shape
        num_heads = atten_weights.shape[0]
        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        x = torch.matmul(x,  atten_weights)
        
        x = x.permute(2, 1, 0, 3).reshape(seq_len, batch_size, -1)
        
        x = x * y
        x = self.out_proj(x)
        return x


class SelfAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, v_head_dim):
        super().__init__()
        self.in_proj = nn.Linear(emb_dim, num_heads * v_head_dim, bias=True)
        self.out_proj = nn.Linear(num_heads * v_head_dim, emb_dim, bias=True)

    def forward(self, x: Tensor, atten_weights: Tensor):
        (seq_len, batch_size, emb_dim) = x.shape
        num_heads = atten_weights.shape[0]
        x = self.in_proj(x)

        x = x.reshape(seq_len, batch_size, num_heads, -1).permute(2, 1, 0, 3)
        x = torch.matmul(x, atten_weights)
        value_head_dim = x.shape[-1]

        x = (
            x.permute(2, 1, 0, 3)
            .contiguous()
            .view(seq_len, batch_size, num_heads * value_head_dim)
        )

        x = self.out_proj(x)
        return x


"""Relative Positional Multi-Head Attention can be implemented 
3x -> chunk & calculate attention  their outputs."""


class RelativePositionalMultiHeadAttention(nn.Module):
    def __init__(
        self,
        emb_dim,
        num_heads,
        q_head_dim,
        pos_head_dim,
        pos_emb_skip_rate: FloatLike = ScheduledFloat([(0, 0.5), (4000, 0)]),
    ):
        super().__init__()
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        self.query_head_dim = q_head_dim
        k_head_dim = q_head_dim
        self.pos_head_dim = pos_head_dim
        self.pos_emb_skip_rate = copy.deepcopy(pos_emb_skip_rate)

        d_out_dim = (emb_dim + k_head_dim + pos_head_dim) // num_heads

        self.in_proj = nn.Linear(emb_dim, d_out_dim)

    def scaled_dot_product_attention(self, q, k, v):
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.query_head_dim**0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output

    def forward(self, x):
        x = self.in_proj(x)
        seq_len, batch_size, d_in = x.shape
        out = self.query_head_dim * self.num_heads

        q = x[..., 0:out]
        k = x[..., out : 2 * out]
        p = x[..., 2 * out :]

        q = q.reshape(seq_len, batch_size, self.num_heads, self.query_head_dim)
        k = k.reshape(seq_len, batch_size, self.num_heads, self.query_head_dim)
        p = p.reshape(seq_len, batch_size, self.num_heads, self.pos_head_dim)

        q = q.permute(2, 1, 0, 3)
        k = k.permute(2, 1, 3, 0)
        p = p.permute(2, 1, 0, 3)

        attn_scores = torch.matmul(q, k)

        use_pos_scores = None
        if not self.training or random.random() >= float(self.pos_emb_skip_rate):
            use_pos_scores = True

        if use_pos_scores:
            pos_scores = torch.matmul(q, p.transpose(-2, -1))
            attn_scores += pos_scores

        attn_weights = torch.softmax(attn_scores, dim=-1)
        return attn_weights


"""
Channel wise scalar
"""


class ByPass(nn.Module):
    def __init__(
        self, emb_dim: int, skip_rate: float = 0.0, straight_through_rate: float = 0.0
    ):
        super().__init__()
        self.bypass_scale = nn.Parameter(torch.full((emb_dim), 0.5))
        self.skip_rate = skip_rate
        self.straight_through_rate = straight_through_rate

    def _get_bypass_scalar(self, batch_size):
        if not self.training:
            return self.bypass_scale
        else:
            ans = limit_param_value()
            if self.skip_rate != 0:
                mask = torch.rand((batch_size, 1)) > self.skip_rate
                ans = ans * mask.float()

            if self.straight_through_rate != 0:
                mask = torch.rand((batch_size, 1)) < self.straight_through_rate
                ans = torch.maximum(mask, ans)

    """
    the 'c' can be different according to the config.
    In Paper it says initialy it is [0.9, 1.0] & change the minimum to 0.2 after 20000 steps"""

    def forward(self, x, c):
        bypass_scalar = self._get_bypass_scalar(x.size(0))
        return bypass_scalar * x + (1 - bypass_scalar) * c


# class BiasNorm(nn.LayerNorm):


def convolution():
    nn.Sequential(
        conv_2d(in_channels=1, out_channels=128, kernel_size=3, padding=1, stride=1),
        nn.ReLU(),
    )


def limit_param_value(x: torch.Tensor, prob: float, trainning: bool):
    if trainning and random.random() < prob:
        return LimitParamValue.apply()
    else:
        return x


class LimitParamValue(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, min: float, max: float):
        ctx.save_for_backward(x)
        ctx.min = min
        ctx.max = max

    """
    Gradient flipping for constrained optimization
    """

    @staticmethod
    def backward(ctx, x_grad: Tensor):
        (x,) = ctx.saved_tensors

        """
        when x is less than the minimum
        """
        x_grad *= torch.where(torch.logical_and(x_grad > 0, x < ctx.min), -1.0, 1.0)

        """
        when x is greater than the maximum
        """
        x_grad *= torch.where(torch.logical_and(x_grad < 0, x > ctx.max), -1.0, 1.0)
        return x_grad, None, None


class Convolution(nn.Module):
    def __init__(self, channels, kernel_size=3, padding=1, stride=1):
        super().__init__()
        self.in_proj = nn.Linear(channels, 2 * channels)
        self.sigmoid = nn.Sigmoid()

        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            padding=kernel_size // 2,
            groups=channels,
        )
        self.out_proj = nn.Linear(channels, channels)

    def forward(self, x):

        # x -> (time, batch, channels)
        x = self.in_proj(x)

        x, y = x.chunk(2, dim=-1)
        x = x * self.sigmoid(y)

        x = x.permute(1, 2, 0)  # (batch, channels, time)
        x = self.depthwise_conv(x)
        x = x.permute(2, 0, 1)

        x = self.out_proj(x)
        return x
