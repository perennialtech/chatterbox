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

    def set_target_voice(self, wav_fpath: str | Path) -> None: ...

    def generate(
        self,
        audio,
        target_voice_path: str | Path | None = None,
        profile: bool = False,
        upscale: bool = False,
    ) -> VCResult: ...
