from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch

from ..models.s3gen.conditioning import S3ReferenceCondition


@dataclass
class VCResult:
    wav: torch.Tensor
    sample_rate: int
    timings: dict[str, float]


class VCBackend(Protocol):
    sr: int

    def set_target_voice_condition(
        self,
        target_voice: dict | S3ReferenceCondition,
    ) -> None: ...

    def convert_from_path(
        self,
        audio_path: str | Path,
        target_voice_path: str | Path | None = None,
        profile: bool = False,
    ) -> VCResult: ...

    def convert_from_tensors(
        self,
        audio_16k: torch.Tensor,
        target_voice: dict | S3ReferenceCondition | None = None,
        profile: bool = False,
    ) -> VCResult: ...
