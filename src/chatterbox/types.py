from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

import numpy as np
import torch

AudioInput: TypeAlias = (
    str
    | Path
    | np.ndarray
    | torch.Tensor
    | tuple[np.ndarray, int]
    | tuple[torch.Tensor, int]
)


@dataclass(frozen=True)
class LoadedAudio:
    waveform: torch.Tensor
    sample_rate: int


@dataclass(frozen=True)
class TokenBatch:
    tokens: torch.LongTensor
    lengths: torch.LongTensor


@dataclass(frozen=True)
class MelBatch:
    mels: torch.Tensor
    lengths: torch.LongTensor


@dataclass(frozen=True)
class ReferenceConditioning:
    prompt_tokens: torch.LongTensor
    prompt_token_lengths: torch.LongTensor
    prompt_mels: torch.Tensor
    prompt_mel_lengths: torch.LongTensor
    speaker_embedding: torch.Tensor


@dataclass(frozen=True)
class ConversionResult:
    waveform: torch.Tensor
    sample_rate: int
    timings: dict[str, float]
