import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _dft_angles(n_fft: int) -> torch.Tensor:
    freq_bins = n_fft // 2 + 1
    k = torch.arange(freq_bins, dtype=torch.float32).unsqueeze(1)
    n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
    return 2.0 * math.pi * k * n / n_fft


class RealSTFT(nn.Module):
    def __init__(self, n_fft: int, hop_len: int, center: bool = True):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.center = center
        angles = _dft_angles(n_fft)
        window = torch.hann_window(n_fft, periodic=True)
        real = torch.cos(angles) * window
        imag = -torch.sin(angles) * window
        basis = torch.cat([real, imag], dim=0).unsqueeze(1)
        self.register_buffer("basis", basis, persistent=False)
        self.freq_bins = n_fft // 2 + 1

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if self.center:
            x = F.pad(x, (self.n_fft // 2, self.n_fft // 2), mode="reflect")
        spec = F.conv1d(
            x, self.basis.to(dtype=x.dtype, device=x.device), stride=self.hop_len
        )
        return spec[:, : self.freq_bins], spec[:, self.freq_bins :]


class RealISTFT(nn.Module):
    def __init__(
        self, n_fft: int, hop_len: int, center: bool = True, eps: float = 1e-8
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_len = hop_len
        self.center = center
        self.eps = eps
        freq_bins = n_fft // 2 + 1
        angles = _dft_angles(n_fft)
        window = torch.hann_window(n_fft, periodic=True)

        factors = torch.ones(freq_bins, dtype=torch.float32)
        if n_fft % 2 == 0:
            factors[1:-1] = 2.0
        else:
            factors[1:] = 2.0
        factors = factors.unsqueeze(1) / n_fft

        real_basis = torch.cos(angles) * factors * window
        imag_basis = -torch.sin(angles) * factors * window
        self.register_buffer("real_basis", real_basis.unsqueeze(1), persistent=False)
        self.register_buffer("imag_basis", imag_basis.unsqueeze(1), persistent=False)
        self.register_buffer(
            "window_sq", window.square().view(1, 1, -1), persistent=False
        )
        self.freq_bins = freq_bins

    def forward(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        real_basis = self.real_basis.to(dtype=magnitude.dtype, device=magnitude.device)
        imag_basis = self.imag_basis.to(dtype=magnitude.dtype, device=magnitude.device)

        wav = F.conv_transpose1d(real, real_basis, stride=self.hop_len)
        wav = wav + F.conv_transpose1d(imag, imag_basis, stride=self.hop_len)

        frames = magnitude.new_ones(magnitude.size(0), 1, magnitude.size(2))
        envelope = F.conv_transpose1d(
            frames,
            self.window_sq.to(dtype=magnitude.dtype, device=magnitude.device),
            stride=self.hop_len,
        )
        wav = wav / envelope.clamp_min(self.eps)

        if self.center:
            pad = self.n_fft // 2
            wav = wav[:, :, pad:-pad]

        return wav.squeeze(1)
