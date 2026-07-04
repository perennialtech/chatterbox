from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Precision = Literal["fp32", "fp16", "both"]
SinglePrecision = Literal["fp32", "fp16"]


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_dir: Path
    output_dir: Path
    opset: int = 18
    precision: Precision = "fp32"
    external_data: bool = True
    validate: bool = True
    device: str = "cpu"
    max_positional_frames: int = 6144

    @property
    def precisions(self) -> tuple[SinglePrecision, ...]:
        if self.precision == "both":
            return ("fp32", "fp16")
        return (self.precision,)

    def onnx_precision_dir(self, precision: SinglePrecision) -> Path:
        return self.output_dir / "onnx" / precision
