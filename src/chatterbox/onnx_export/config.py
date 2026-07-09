from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_dir: Path
    output_dir: Path
    opset: int = 18
    external_data: bool = True
    validate: bool = True
    device: str = "cpu"

    @property
    def onnx_dir(self) -> Path:
        return self.output_dir / "onnx"
