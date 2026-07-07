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


class MaskedGroupNorm1D(nn.Module):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1.0e-5,
        affine: bool = True,
    ):
        super().__init__()
        if num_groups <= 0:
            raise ValueError(f"num_groups must be positive, got {num_groups}")
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        if num_channels % num_groups != 0:
            raise ValueError(
                f"num_channels must be divisible by num_groups, got {num_channels} and {num_groups}"
            )

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.weight = nn.Parameter(torch.ones(num_channels))
            self.bias = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Normalize x shaped [batch, channels, time] with mask shaped [batch, 1, time]."""
        _validate_mask_1d(x, mask)

        if x.shape[1] != self.num_channels:
            raise ValueError(
                f"x channel dimension must be {self.num_channels}, got {x.shape[1]}"
            )
        if not x.is_floating_point():
            raise TypeError(f"x must be floating point, got {x.dtype}")

        batch, channels, time = x.shape
        channels_per_group = channels // self.num_groups
        stats_dtype = (
            torch.float32 if x.dtype in (torch.float16, torch.bfloat16) else x.dtype
        )

        x_stats = x.to(dtype=stats_dtype)
        mask_stats = mask.to(dtype=stats_dtype)

        x_grouped = x_stats.reshape(batch, self.num_groups, channels_per_group, time)
        valid = mask_stats.reshape(batch, 1, 1, time)

        denom = (valid.sum(dim=-1, keepdim=True) * channels_per_group).clamp_min(1.0)
        mean = (x_grouped * valid).sum(dim=(2, 3), keepdim=True) / denom

        centered = (x_grouped - mean) * valid
        var = centered.square().sum(dim=(2, 3), keepdim=True) / denom

        output = (x_grouped - mean) * torch.rsqrt(var + self.eps)
        output = output.reshape(batch, channels, time).to(dtype=x.dtype)

        if self.affine:
            output = output * self.weight.view(1, channels, 1) + self.bias.view(
                1, channels, 1
            )

        return output * mask.to(dtype=output.dtype)

    def extra_repr(self) -> str:
        return (
            f"{self.num_groups}, {self.num_channels}, "
            f"eps={self.eps}, affine={self.affine}"
        )


class _MaskedConvNormAct1D(nn.Sequential):
    def __init__(self, dim: int, dim_out: int, groups: int):
        super().__init__(
            nn.Conv1d(dim, dim_out, 3, padding=1),
            MaskedGroupNorm1D(groups, dim_out),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply conv, masked norm, and activation to [batch, channels, time] input."""
        output = self[0](x * mask)
        output = self[1](output, mask)
        return self[2](output)


class Block1D(nn.Module):
    def __init__(self, dim: int, dim_out: int, groups: int = 8):
        super().__init__()
        self.block = _MaskedConvNormAct1D(dim, dim_out, groups)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Map x shaped [batch, dim, time] to [batch, dim_out, time] using mask [batch, 1, time]."""
        _validate_mask_1d(x, mask)
        output = self.block(x, mask)
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
        """Map x [batch, dim, time] to [batch, dim_out, time] with mask [batch, 1, time] and time_emb [batch, time_emb_dim]."""
        _validate_mask_1d(x, mask)
        if time_emb.ndim != 2:
            raise ValueError(
                f"time_emb must have shape [batch, time_emb_dim], got {tuple(time_emb.shape)}"
            )
        if time_emb.shape[0] != x.shape[0]:
            raise ValueError(
                f"time_emb batch size must match x batch size, got {time_emb.shape[0]} and {x.shape[0]}"
            )

        h = self.block1(x, mask)
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        return (h + self.res_conv(x * mask)) * mask


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
