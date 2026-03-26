import torch
import torch.nn as nn
import random
from torch import Tensor

from zipvoice.zipformer.biasnorm import BiasNorm


class Zipformer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        encoder_dim=384,
        num_encoder_layers: int = 4,
    ):
        super(Zipformer, self).__init__()
        self.conv_emb = conv_embedding(self, in_channels=in_dim, out_channels=out_dim)

        self.in_proj = nn.Linear(in_dim, encoder_dim)
        self.out_proj = nn.Linear(encoder_dim, out_dim)

        encoder_layer = []
        encoder_layer.append(ZipformerBlock())

        for _ in range(num_encoder_layers):
            zipformer_block = ZipformerBlock()
            encoder = ZipformerEcoder(zipformer_block)
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


class ZipformerEcoder(nn.Module):
    def __init__(self, encoder_layers):
        super().__init__()
        self.encoder_layer = encoder_layers

    def forward(self, x):
        for layer in self.encoder_layer:
            x = layer(x)
        return x


class DownsampledZipformerEncoder(nn.Module):
    def __init__(self, encoder_layer):
        super().__init__()
        self.encoder_layer = encoder_layer
        self.downsample = Downsample(downsample_factor=2)
        self.upsample = Upsample(upsample_factor=2)
        self.bypass = ByPass(emb_dim=512, skip_rate=0.1, straight_through_rate=0.1)

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


# class ZipformerWrapper(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
#         self.zipformer = Zipformer()
#         self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)

#     def forward(self, x):
#         return self.zipformer(x)


def conv_embedding(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Sequential(
        conv_2d(
            in_channels=in_channels,
            out_channels=out_channels // 16,
            kernel_size=kernel_size,
            padding=padding,
            stride=(1, 2),
        ),
        nn.ReLU(),
        conv_2d(
            in_channels=in_channels,
            out_channels=out_channels // 4,
            kernel_size=kernel_size,
            padding=padding,
            stride=(2, 2),
        ),
        nn.ReLU(),
        conv_2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=(1, 2),
        ),
    )


def conv_2d(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        padding=padding,
        stride=stride,
    )


class ZipformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.feed_forward_1 = ZipformerFeedForward()
        self.non_linear_attention = NonLinearAttention()
        self.self_attention_1 = RelativePositionalMultiHeadAttention(
            d_in=512, d_out=512, d_v_out=512
        )
        self.convolution_1 = convolution()

        self.feed_forward_2 = ZipformerFeedForward()
        self.bypass_1 = ByPass(emb_dim=512, skip_rate=0.1, straight_through_rate=0.1)
        self.self_attention_2 = SelfAttention(d_in=512, d_out=512, d_v_out=512)
        self.convolution_2 = convolution()
        self.feed_forward_3 = ZipformerFeedForward()
        self.bias_norm = BiasNorm(512)
        self.bypass_2 = ByPass(emb_dim=512, skip_rate=0.1, straight_through_rate=0.1)

    def forward(self, x, atten_masks=None):
        atten_weights = self.self_attention_1(x)

        x = self.feed_forward_1(x)
        x = self.non_linear_attention(x, atten_weights)
        x = self.self_attention_1(x, atten_weights)

        # conv

        x = self.feed_forward_2(x)
        x = self.bypass_1(x, c)
        x = self.self_attention_2(x)
        # conv
        x = self.feed_forward_3(x)

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
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(512, 512 * 3)
        self.tanh = nn.Tanh()
        self.relative_positional_multi_head_atten = (
            RelativePositionalMultiHeadAttention(
                d_in=512, d_out=512, num_heads=1, q_head_dim=512, pos_head_dim=512
            )
        )
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x, single_head_atten_weights=None):
        x = self.in_proj(x)
        s, y, z = x.chunk(3, dim=-1)
        s = self.tanh(s)
        s *= y
        s = self.single_head_atten(s)
        s *= single_head_atten_weights
        s *= z

        s = self.out_proj(s)
        return s


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, d_v_out):
        super().__init__()
        self.q = nn.Linear(d_in, d_out)
        self.k = nn.Linear(d_in, d_out)
        self.v = nn.Linear(d_in, d_v_out)

    def forward(self, x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (512**0.5)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, v)
        return output


"""Multi-Head Attention can be implemented by creating multiple instances 
of SelfAttention and concatenating their outputs."""


class RelativePositionalMultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, q_head_dim, pos_head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.d_in = d_in
        self.d_out = d_out
        self.query_head_dim = q_head_dim
        k_head_dim = q_head_dim
        self.pos_head_dim = pos_head_dim

        d_out_dim = (d_out + k_head_dim + pos_head_dim) // num_heads

        self.in_proj = nn.Linear(d_in, d_out_dim)

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
