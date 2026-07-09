from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..models.s3gen import S3Gen


@dataclass(frozen=True)
class ExportContext:
    source_hop: int
    vocoder_harmonics: int
    token_mel_ratio: int
    device: str | torch.device = "cpu"
    dtype: torch.dtype = torch.float32

    @classmethod
    def from_model(
        cls,
        model: S3Gen,
        *,
        device: str | torch.device,
        dtype: torch.dtype | None = None,
    ) -> "ExportContext":
        if dtype is None:
            dtype = next(model.parameters()).dtype
        return cls(
            source_hop=int(model.mel2wav.source_hop),
            vocoder_harmonics=int(model.mel2wav.nb_harmonics + 1),
            token_mel_ratio=int(model._token_mel_ratio),
            device=device,
            dtype=dtype,
        )


@dataclass(frozen=True)
class GraphSpec:
    name: str
    filename: str
    input_names: list[str]
    output_names: list[str]
    dynamic_shapes: dict[str, Any] | tuple[Any, ...]
    make_module: Callable[[S3Gen], torch.nn.Module]
    make_dummy_inputs: Callable[..., tuple[torch.Tensor, ...]]
    input_dtypes: dict[str, str]
    output_dtypes: dict[str, str]
    required_for_runtime: bool = True
