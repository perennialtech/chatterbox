from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torchaudio as ta

from .device import Runtime
from .types import AudioInput, LoadedAudio


@lru_cache(maxsize=64)
def _cached_resampler(src_sr: int, dst_sr: int, device: str) -> ta.transforms.Resample:
    return ta.transforms.Resample(src_sr, dst_sr).to(torch.device(device))


class AudioProcessor:
    def __init__(self, runtime: Runtime):
        self.runtime = runtime

    def load(
        self,
        audio: AudioInput,
        *,
        sample_rate: int,
        max_samples: int | None = None,
    ) -> torch.Tensor:
        loaded = self.load_native(audio)
        wav = self.to_mono_batch(loaded.waveform)
        wav = wav.to(device=self.runtime.device, dtype=torch.float32)

        if loaded.sample_rate != sample_rate:
            wav = self.resample(wav, loaded.sample_rate, sample_rate)

        if max_samples is not None:
            wav = wav[:, :max_samples]

        return wav.contiguous()

    def load_native(self, audio: AudioInput) -> LoadedAudio:
        if isinstance(audio, tuple):
            wav, sr = audio
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            return LoadedAudio(wav.float(), int(sr))

        if isinstance(audio, np.ndarray):
            raise ValueError(
                "Raw NumPy audio must be passed as `(array, sample_rate)`."
            )

        if torch.is_tensor(audio):
            raise ValueError(
                "Raw tensor audio must be passed as `(tensor, sample_rate)`."
            )

        wav, sr = ta.load(str(Path(audio).expanduser()))
        return LoadedAudio(wav.float(), int(sr))

    def to_mono_batch(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.ndim == 1:
            return wav.unsqueeze(0)
        if wav.ndim != 2:
            raise ValueError("Audio must have shape [samples] or [channels, samples].")
        if wav.size(0) == 1:
            return wav
        return wav.mean(dim=0, keepdim=True)

    def resample(self, wav: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
        if src_sr == dst_sr:
            return wav

        resampler = _cached_resampler(src_sr, dst_sr, str(self.runtime.device))
        return resampler(wav)
