import math

import librosa
import torch
import torch.nn.functional as F

from ...audio import S3GEN_SR


class S3GenMelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 1920,
        num_mels: int = 80,
        sampling_rate: int = S3GEN_SR,
        hop_size: int = 480,
        win_size: int = 1920,
        fmin: int = 0,
        fmax: int = 8000,
    ):
        super().__init__()
        if win_size != n_fft:
            raise ValueError("S3GenMelSpectrogram requires win_size == n_fft")

        self.n_fft = n_fft
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.pad = int((n_fft - hop_size) / 2)
        self.freq_bins = n_fft // 2 + 1

        n = torch.arange(n_fft, dtype=torch.float32).unsqueeze(0)
        k = torch.arange(self.freq_bins, dtype=torch.float32).unsqueeze(1)
        angles = 2.0 * math.pi * k * n / n_fft
        window = torch.hann_window(win_size, periodic=True)
        real = torch.cos(angles) * window
        imag = -torch.sin(angles) * window
        basis = torch.cat([real, imag], dim=0).unsqueeze(1)
        self.register_buffer("stft_basis", basis, persistent=False)

        mel = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.register_buffer(
            "mel_basis", torch.from_numpy(mel).float(), persistent=False
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        if wav.ndim != 2:
            raise ValueError(
                f"wav must have shape [T] or [B, T], got {tuple(wav.shape)}"
            )

        wav = wav.to(dtype=self.stft_basis.dtype, device=self.stft_basis.device)
        x = F.pad(wav.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        spec = F.conv1d(
            x,
            self.stft_basis.to(dtype=x.dtype, device=x.device),
            stride=self.hop_size,
        )
        real = spec[:, : self.freq_bins]
        imag = spec[:, self.freq_bins :]
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-9)
        mel = torch.matmul(
            self.mel_basis.to(dtype=magnitude.dtype, device=magnitude.device),
            magnitude,
        )
        return torch.log(torch.clamp(mel, min=1e-5))
