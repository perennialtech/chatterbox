import logging

import numpy as np
import torch
from librosa.filters import mel as librosa_mel_fn

logger = logging.getLogger(__name__)


def dynamic_range_compression_torch(
    x: torch.Tensor,
    C: float = 1.0,
    clip_val: float = 1e-5,
) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=clip_val) * C)


def spectral_normalize_torch(magnitudes: torch.Tensor) -> torch.Tensor:
    return dynamic_range_compression_torch(magnitudes)


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        n_fft: int = 1920,
        num_mels: int = 80,
        sampling_rate: int = 24000,
        hop_size: int = 480,
        win_size: int = 1920,
        fmin: int = 0,
        fmax: int = 8000,
        center: bool = False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.sampling_rate = sampling_rate
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax
        self.center = center

        mel = librosa_mel_fn(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())
        self.register_buffer("hann_window", torch.hann_window(win_size))

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        if y.ndim == 1:
            y = y.unsqueeze(0)

        y = y.to(dtype=self.mel_basis.dtype)
        min_val = torch.min(y)
        max_val = torch.max(y)
        if min_val < -1.0 or max_val > 1.0:
            logger.warning(
                "Audio values outside normalized range: min=%.4f, max=%.4f",
                float(min_val.detach().cpu()),
                float(max_val.detach().cpu()),
            )

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_size) / 2),
                int((self.n_fft - self.hop_size) / 2),
            ),
            mode="reflect",
        ).squeeze(1)

        spec = torch.view_as_real(
            torch.stft(
                y,
                self.n_fft,
                hop_length=self.hop_size,
                win_length=self.win_size,
                window=self.hann_window.to(dtype=y.dtype, device=y.device),
                center=self.center,
                pad_mode="reflect",
                normalized=False,
                onesided=True,
                return_complex=True,
            )
        )
        spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-9)
        spec = torch.matmul(
            self.mel_basis.to(dtype=spec.dtype, device=spec.device), spec
        )
        return spectral_normalize_torch(spec)


_DEFAULT_MEL = MelSpectrogram()


def mel_spectrogram(
    y,
    n_fft: int = 1920,
    num_mels: int = 80,
    sampling_rate: int = 24000,
    hop_size: int = 480,
    win_size: int = 1920,
    fmin: int = 0,
    fmax: int = 8000,
    center: bool = False,
) -> torch.Tensor:
    if isinstance(y, np.ndarray):
        y = torch.from_numpy(y).float()

    default_matches = (
        n_fft == _DEFAULT_MEL.n_fft
        and num_mels == _DEFAULT_MEL.num_mels
        and sampling_rate == _DEFAULT_MEL.sampling_rate
        and hop_size == _DEFAULT_MEL.hop_size
        and win_size == _DEFAULT_MEL.win_size
        and fmin == _DEFAULT_MEL.fmin
        and fmax == _DEFAULT_MEL.fmax
        and center == _DEFAULT_MEL.center
    )
    module = (
        _DEFAULT_MEL
        if default_matches
        else MelSpectrogram(
            n_fft=n_fft,
            num_mels=num_mels,
            sampling_rate=sampling_rate,
            hop_size=hop_size,
            win_size=win_size,
            fmin=fmin,
            fmax=fmax,
            center=center,
        )
    )
    module = module.to(device=y.device)
    return module(y)
