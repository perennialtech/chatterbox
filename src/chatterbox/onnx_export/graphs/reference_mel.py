from __future__ import annotations

import math

import librosa
import torch
import torch.nn.functional as F

from ...audio import S3GEN_SR
from ..constants import GRAPH_REFERENCE_MEL_24K
from ..dynamic_axes import REFERENCE_MEL_DYNAMIC_AXES
from ..graph_spec import GraphSpec
from ..names import REFERENCE_MEL_24K

input_names = ["wav_24k"]
output_names = ["prompt_feat", "prompt_feat_len"]
dynamic_axes = REFERENCE_MEL_DYNAMIC_AXES


class ExportableMelSpectrogram(torch.nn.Module):
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
            raise ValueError(
                "ExportableMelSpectrogram currently requires win_size == n_fft"
            )
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
        self.register_buffer("stft_basis", basis)

        mel = librosa.filters.mel(
            sr=sampling_rate,
            n_fft=n_fft,
            n_mels=num_mels,
            fmin=fmin,
            fmax=fmax,
        )
        self.register_buffer("mel_basis", torch.from_numpy(mel).float())

    def forward(self, wav_24k: torch.Tensor) -> torch.Tensor:
        if wav_24k.ndim == 1:
            wav_24k = wav_24k.unsqueeze(0)
        wav_24k = wav_24k.to(dtype=self.stft_basis.dtype)
        x = F.pad(wav_24k.unsqueeze(1), (self.pad, self.pad), mode="reflect")
        spec = F.conv1d(
            x,
            self.stft_basis.to(dtype=x.dtype, device=x.device),
            stride=self.hop_size,
        )
        real = spec[:, : self.freq_bins]
        imag = spec[:, self.freq_bins :]
        magnitude = torch.sqrt(real.square() + imag.square() + 1e-9)
        mel = torch.matmul(
            self.mel_basis.to(dtype=magnitude.dtype, device=magnitude.device), magnitude
        )
        return torch.log(torch.clamp(mel, min=1e-5))


class ReferenceMel24kExport(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = ExportableMelSpectrogram(sampling_rate=S3GEN_SR)

    def forward(self, wav_24k):
        prompt_feat = self.mel(wav_24k).transpose(1, 2).contiguous()
        prompt_feat_len = torch.full(
            (prompt_feat.size(0),),
            prompt_feat.size(1),
            dtype=torch.long,
            device=prompt_feat.device,
        )
        return prompt_feat, prompt_feat_len


def make_module(model):
    return ReferenceMel24kExport()


def make_dummy_inputs(batch: int = 1, samples: int = S3GEN_SR):
    return (torch.randn(batch, samples, dtype=torch.float32),)


REFERENCE_MEL_24K_SPEC = GraphSpec(
    name=GRAPH_REFERENCE_MEL_24K,
    filename=REFERENCE_MEL_24K,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    make_module=make_module,
    make_dummy_inputs=make_dummy_inputs,
    input_dtypes={"wav_24k": "float32"},
    output_dtypes={"prompt_feat": "float32", "prompt_feat_len": "int64"},
)
