from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch


@dataclass
class VCResult:
    wav: torch.Tensor
    sample_rate: int
    timings: dict[str, float]


class VCBackend(Protocol):
    sr: int

    def set_target_voice_from_tensors(self, target_voice: dict) -> None: ...

    def convert_from_path(
        self,
        audio_path: str | Path,
        target_voice_path: str | Path | None = None,
        profile: bool = False,
        upscale: bool = False,
    ) -> VCResult: ...

    def convert_from_tensors(
        self,
        audio_16k: torch.Tensor,
        target_voice: dict | None = None,
        profile: bool = False,
        upscale: bool = False,
    ) -> VCResult: ...
