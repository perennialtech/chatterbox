import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalPosEmb requires dim to be even")
        if dim < 2:
            raise ValueError("SinusoidalPosEmb requires dim >= 2")

        self.dim = dim
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 1:
            x = x.unsqueeze(0)

        half_dim = self.dim // 2
        exponent = math.log(10000.0) / (half_dim - 1)
        frequencies = torch.exp(
            torch.arange(half_dim, device=x.device, dtype=torch.float32) * -exponent
        )

        emb = self.scale * x.unsqueeze(1) * frequencies.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Block1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        output = self.block(x * mask)
        return output * mask


class ResnetBlock1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(time_emb_dim, dim_out),
        )
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        time_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.block1(x, mask)
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        return h + self.res_conv(x * mask)


class Downsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class TimestepEmbedding(nn.Module):
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: Optional[int] = None,
        post_act_fn: Optional[str] = None,
    ):
        super().__init__()

        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        self.act = self._make_activation(act_fn)

        time_embed_dim_out = out_dim if out_dim is not None else time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)

        self.post_act = (
            None if post_act_fn is None else self._make_activation(post_act_fn)
        )

    @staticmethod
    def _make_activation(name: str) -> nn.Module:
        if name == "silu":
            return nn.SiLU()
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.linear_1(sample)
        sample = self.act(sample)
        sample = self.linear_2(sample)

        if self.post_act is not None:
            sample = self.post_act(sample)

        return sample


class Upsample1D(nn.Module):
    def __init__(
        self,
        channels: int,
        out_channels: Optional[int] = None,
    ):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv(inputs)
