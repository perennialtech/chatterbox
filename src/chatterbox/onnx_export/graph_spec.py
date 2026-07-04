from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import torch

from ..models.s3gen import S3Gen


@dataclass(frozen=True)
class GraphSpec:
    name: str
    filename: str
    input_names: list[str]
    output_names: list[str]
    dynamic_shapes: dict[str, Any] | tuple[Any, ...]
    make_module: Callable[[S3Gen], torch.nn.Module]
    make_dummy_inputs: Callable[[], tuple[torch.Tensor, ...]]
    input_dtypes: dict[str, str]
    output_dtypes: dict[str, str]
    required_for_runtime: bool = True
