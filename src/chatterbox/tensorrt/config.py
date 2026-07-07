from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class TrtBuildConfig:
    artifact_dir: Path
    output_dir: Path | None = None
    engine_precision: Literal["fp32", "fp16"] = "fp32"
    workspace_bytes: int = 4 * 1024**3
    shape_plan: Path | None = None
    strongly_typed: bool = True

    @property
    def resolved_output_dir(self) -> Path:
        return self.output_dir or (
            self.artifact_dir / "tensorrt" / self.engine_precision
        )
