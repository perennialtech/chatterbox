import math
from typing import Optional

import torch
import torch.nn as nn


def _validate_mask_1d(x: torch.Tensor, mask: torch.Tensor) -> None:
    """Validate a sequence mask for an input shaped [batch, channels, time]."""
    if x.ndim != 3:
        raise ValueError(
            f"x must have shape [batch, channels, time], got {tuple(x.shape)}"
        )
    if mask.ndim != 3:
        raise ValueError(
            f"mask must have shape [batch, 1, time], got {tuple(mask.shape)}"
        )
    if mask.shape[0] != x.shape[0]:
        raise ValueError(
            f"mask batch size must match x batch size, got {mask.shape[0]} and {x.shape[0]}"
        )
    if mask.shape[1] != 1:
        raise ValueError(f"mask channel dimension must be 1, got {mask.shape[1]}")
    if mask.shape[2] != x.shape[2]:
        raise ValueError(
            f"mask time dimension must match x time dimension, got {mask.shape[2]} and {x.shape[2]}"
        )


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, scale: float = 1000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("SinusoidalPosEmb requires dim to be even")
        if dim < 2:
            raise ValueError("SinusoidalPosEmb requires dim >= 2")

        self.dim = dim
        self.scale = scale

        half_dim = dim // 2
        if half_dim == 1:
            frequencies = torch.ones(1, dtype=torch.float32)
        else:
            exponent = math.log(10000.0) / (half_dim - 1)
            frequencies = torch.exp(
                torch.arange(half_dim, dtype=torch.float32) * -exponent
            )

        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed scalar or [batch] timesteps into [batch, dim] sinusoidal features."""
        if x.ndim == 0:
            x = x.unsqueeze(0)
        elif x.ndim != 1:
            raise ValueError(
                f"SinusoidalPosEmb expects a scalar tensor or shape [batch], got {tuple(x.shape)}"
            )

        frequencies = self.frequencies.to(device=x.device, dtype=torch.float32)

        emb = (
            self.scale
            * x.to(dtype=torch.float32).unsqueeze(1)
            * frequencies.unsqueeze(0)
        )
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Downsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample x shaped [batch, channels, time] to [batch, channels, ceil(time / 2)]."""
        if x.ndim != 3:
            raise ValueError(
                f"x must have shape [batch, channels, time], got {tuple(x.shape)}"
            )
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
        if name == "mish":
            return nn.Mish()
        raise ValueError(f"Unsupported activation function: {name}")

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        """Project sample shaped [batch, in_channels] to [batch, out_dim or time_embed_dim]."""
        if sample.ndim != 2:
            raise ValueError(
                f"sample must have shape [batch, in_channels], got {tuple(sample.shape)}"
            )

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
        """Upsample inputs shaped [batch, channels, time] to [batch, out_channels, 2 * time]."""
        if inputs.ndim != 3:
            raise ValueError(
                f"inputs must have shape [batch, channels, time], got {tuple(inputs.shape)}"
            )
        return self.conv(inputs)
