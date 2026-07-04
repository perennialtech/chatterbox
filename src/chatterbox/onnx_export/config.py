from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ExportProfile = Literal["vc_minimal", "vc_reference", "vc_full_tensor", "vc_bucketed"]
Precision = Literal["fp32", "fp16"]
Quantization = Literal["none", "int8", "q4", "q4f16"]


@dataclass(frozen=True)
class ExportConfig:
    checkpoint_dir: Path
    output_dir: Path
    profile: ExportProfile = "vc_minimal"
    opset: int = 18
    precision: Precision = "fp32"
    quantization: Quantization = "none"
    external_data: bool = True
    buckets: tuple[int, ...] = (384, 512, 768, 1024)
    validate: bool = True
    device: str = "cpu"

    @property
    def precision_dir(self) -> Path:
        return self.output_dir / self.precision
