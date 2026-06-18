from __future__ import annotations

import librosa
import torch


class MelSpectrogram(torch.nn.Module):
    def __init__(
        self,
        *,
        sample_rate: int = 24_000,
        n_fft: int = 1920,
        num_mels: int = 80,
        hop_size: int = 480,
        win_size: int = 1920,
        fmin: int = 0,
        fmax: int = 8000,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.num_mels = num_mels
        self.hop_size = hop_size
        self.win_size = win_size
        self.fmin = fmin
        self.fmax = fmax

        mel = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.register_buffer("mel_basis", torch.tensor(mel, dtype=torch.float32))
        self.register_buffer("window", torch.hann_window(win_size))

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim != 2:
            raise ValueError("Mel input must have shape [B, samples].")

        wav = torch.nn.functional.pad(
            wav.unsqueeze(1),
            ((self.n_fft - self.hop_size) // 2, (self.n_fft - self.hop_size) // 2),
            mode="reflect",
        ).squeeze(1)

        spec = torch.stft(
            wav,
            self.n_fft,
            hop_length=self.hop_size,
            win_length=self.win_size,
            window=self.window.to(device=wav.device, dtype=wav.dtype),
            center=False,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        mag = torch.sqrt(spec.real.square() + spec.imag.square() + 1.0e-9)
        mel = self.mel_basis.to(device=wav.device, dtype=wav.dtype) @ mag
        return torch.log(torch.clamp(mel, min=1.0e-5))
