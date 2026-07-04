from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..onnx_export.artifacts import manifest_hash
from .cuda import cuda_runtime_version
from .types import ShapeRange


@dataclass
class EngineRecord:
    graph_name: str
    engine: str
    source_onnx: str
    source_onnx_hash: str
    inputs: list[str]
    outputs: list[str]
    shape_ranges: dict[str, ShapeRange]


def write_trt_manifest(
    engine_dir: Path,
    source_manifest: Path,
    precision: str,
    records: list[EngineRecord],
    constants: dict,
) -> None:
    import tensorrt as trt

    engine_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "source_manifest_hash": manifest_hash(source_manifest),
        "tensorrt_version": trt.__version__,
        "cuda_runtime_version": cuda_runtime_version(),
        "precision": precision,
        "constants": constants,
        "engines": {
            record.graph_name: {
                "engine": record.engine,
                "source_onnx": record.source_onnx,
                "source_onnx_hash": record.source_onnx_hash,
                "inputs": record.inputs,
                "outputs": record.outputs,
                "shape_ranges": {
                    name: shape.as_dict() for name, shape in record.shape_ranges.items()
                },
            }
            for record in records
        },
    }
    (engine_dir / "trt_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )


def load_trt_manifest(engine_dir: Path) -> dict:
    path = Path(engine_dir) / "trt_manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing TensorRT manifest: {path}")
    return json.loads(path.read_text())
