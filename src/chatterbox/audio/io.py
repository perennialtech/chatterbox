from pathlib import Path

import torch
import torchaudio as ta

from .resample import resample_audio


def load_audio_mono(
    path: str | Path,
    sample_rate: int,
    device,
    max_len: int | None = None,
) -> torch.Tensor:
    wav, src_sr = ta.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    wav = resample_audio(wav, src_sr, sample_rate, device).squeeze(0)

    if max_len is not None:
        wav = wav[:max_len]

    return wav.contiguous()
