# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Zhihao Du)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import (Block1D, Downsample1D, ResnetBlock1D, SinusoidalPosEmb,
                     TimestepEmbedding, Upsample1D)
from .transformer_block import BasicTransformerBlock
from .utils.intmeanflow import get_intmeanflow_time_mixer
from .utils.mask import add_optional_chunk_mask


def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if mask.dtype != torch.bool:
        raise TypeError("attention mask must be bool before bias conversion")
    neg = -1.0e4 if dtype in (torch.float16, torch.bfloat16) else -1.0e9
    zeros = torch.zeros((), dtype=dtype, device=mask.device)
    negs = torch.full((), neg, dtype=dtype, device=mask.device)
    return torch.where(mask, zeros, negs)


class Transpose(torch.nn.Module):
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x: torch.Tensor):
        x = torch.transpose(x, self.dim0, self.dim1)
        return x


class CausalBlock1D(Block1D):
    def __init__(self, dim: int, dim_out: int):
        super(CausalBlock1D, self).__init__(dim, dim_out)
        self.block = torch.nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        output = self.block(x * mask)
        return output * mask


class CausalResnetBlock1D(ResnetBlock1D):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super(CausalResnetBlock1D, self).__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)


class CausalConv1d(torch.nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
            device=device,
            dtype=dtype,
        )
        assert stride == 1
        self.causal_padding = (kernel_size - 1, 0)

    def forward(self, x: torch.Tensor):
        x = F.pad(x, self.causal_padding)
        x = super(CausalConv1d, self).forward(x)
        return x


class ConditionalDecoder(nn.Module):
    def __init__(
        self,
        in_channels=320,
        out_channels=80,
        channels=[256],
        dropout=0.0,
        attention_head_dim=64,
        n_blocks=4,
        num_mid_blocks=12,
        num_heads=8,
        act_fn="gelu",
        meanflow=False,
    ):
        super().__init__()
        channels = tuple(channels)
        self.meanflow = meanflow
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        time_embed_dim = channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=time_embed_dim,
            act_fn="silu",
        )

        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        self.static_chunk_size = 0

        output_channel = in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = CausalResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            downsample = (
                Downsample1D(output_channel)
                if not is_last
                else CausalConv1d(output_channel, output_channel, 3)
            )
            self.down_blocks.append(
                nn.ModuleList([resnet, transformer_blocks, downsample])
            )

        for _ in range(num_mid_blocks):
            input_channel = channels[-1]
            resnet = CausalResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )

            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )

            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))

        channels = channels[::-1] + (channels[0],)
        for i in range(len(channels) - 1):
            input_channel = channels[i] * 2
            output_channel = channels[i + 1]
            is_last = i == len(channels) - 2
            resnet = CausalResnetBlock1D(
                dim=input_channel,
                dim_out=output_channel,
                time_emb_dim=time_embed_dim,
            )
            transformer_blocks = nn.ModuleList(
                [
                    BasicTransformerBlock(
                        dim=output_channel,
                        num_attention_heads=num_heads,
                        attention_head_dim=attention_head_dim,
                        dropout=dropout,
                        activation_fn=act_fn,
                    )
                    for _ in range(n_blocks)
                ]
            )
            upsample = (
                Upsample1D(output_channel)
                if not is_last
                else CausalConv1d(output_channel, output_channel, 3)
            )
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        self.final_block = CausalBlock1D(channels[-1], channels[-1])
        self.final_proj = nn.Conv1d(channels[-1], self.out_channels, 1)
        self.initialize_weights()
        self.time_embed_mixer = None
        if self.meanflow:
            self.time_embed_mixer = get_intmeanflow_time_mixer(time_embed_dim)

    @property
    def dtype(self):
        return self.final_proj.weight.dtype

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks, cond, r):
        t = self.time_embeddings(t).to(dtype=x.dtype)
        t = self.time_mlp(t)

        if self.meanflow:
            r = self.time_embeddings(r).to(dtype=x.dtype)
            r = self.time_mlp(r)
            t = self.time_embed_mixer(torch.cat([t, r], dim=1))

        x = torch.cat([x, mu], dim=1)

        spks = spks.unsqueeze(-1).expand(-1, -1, x.size(-1))
        x = torch.cat([x, spks, cond], dim=1)

        attn_mask = add_optional_chunk_mask(mask.bool())
        attn_bias = mask_to_bias(attn_mask, x.dtype)

        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t)
            x = x.transpose(1, 2).contiguous()
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_bias,
                )
            x = x.transpose(1, 2).contiguous()
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]

        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t)
            x = x.transpose(1, 2).contiguous()
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_bias,
                )
            x = x.transpose(1, 2).contiguous()

        mask_up = masks[-1]
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = torch.cat([x[:, :, : skip.shape[-1]], skip], dim=1)
            x = resnet(x, mask_up, t)
            x = x.transpose(1, 2).contiguous()
            for transformer_block in transformer_blocks:
                x = transformer_block(
                    hidden_states=x,
                    attention_mask=attn_bias,
                )
            x = x.transpose(1, 2).contiguous()
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask

    def forward_export(self, x, mask, mu, spks, cond, t, r):
        return self.forward(x, mask, mu, t, spks, cond, r)
