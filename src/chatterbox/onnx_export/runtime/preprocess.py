from pathlib import Path

import torch

from ...audio import S3GEN_SR, load_audio_mono


def load_reference_wav(
    path: str | Path, device: str = "cpu", max_len: int | None = None
) -> torch.Tensor:
    return load_audio_mono(path, S3GEN_SR, device, max_len=max_len).unsqueeze(0)
