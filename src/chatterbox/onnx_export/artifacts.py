from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..audio import (DEC_COND_LEN, ENC_COND_LEN, MEL_HOP_24K, S3_HOP, S3_SR,
                     S3_TOKEN_HOP, S3_TOKEN_RATE, S3GEN_SR, SPEECH_VOCAB_SIZE)
from .config import ExportConfig, SinglePrecision
from .constants import MEANFLOW_T_SPAN
from .graph_spec import GraphSpec


@dataclass
class ArtifactRecord:
    graph_name: str
    precision: SinglePrecision
    path: str
    inputs: list[str]
    outputs: list[str]
    dynamic_axes: dict[str, Any]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_hash(checkpoint_dir: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(checkpoint_dir.glob("*")):
        if path.is_file():
            h.update(path.name.encode())
            h.update(sha256_file(path).encode())
    return h.hexdigest()


def manifest_hash(path: Path) -> str:
    return sha256_file(path)


def _constants(source_hop: int) -> dict[str, Any]:
    n_trim = S3GEN_SR // 50
    return {
        "sample_rates": {"s3": S3_SR, "s3gen": S3GEN_SR},
        "hop_sizes": {"s3": S3_HOP, "s3_token": S3_TOKEN_HOP, "mel_24k": MEL_HOP_24K},
        "token_rate": S3_TOKEN_RATE,
        "speech_vocab_size": SPEECH_VOCAB_SIZE,
        "source_hop": source_hop,
        "meanflow_t_span": list(MEANFLOW_T_SPAN),
        "trim_fade_len": 2 * n_trim,
        "prompt_limits": {
            "encoder_samples": ENC_COND_LEN,
            "decoder_samples": DEC_COND_LEN,
        },
    }


def write_manifest(
    output_dir: Path,
    config: ExportConfig,
    graph_specs: tuple[GraphSpec, ...],
    artifacts: list[ArtifactRecord],
    source_hop: int,
    ort_version: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_map: dict[str, dict[str, ArtifactRecord]] = {}
    for artifact in artifacts:
        artifact_map.setdefault(artifact.graph_name, {})[artifact.precision] = artifact

    graphs: dict[str, Any] = {}
    for spec in graph_specs:
        precision_files = {}
        for precision in config.precisions:
            artifact = artifact_map.get(spec.name, {}).get(precision)
            if artifact is not None:
                precision_files[precision] = str(
                    Path(artifact.path).relative_to(output_dir)
                )

        graphs[spec.name] = {
            "filename": spec.filename,
            "files": precision_files,
            "inputs": spec.input_names,
            "outputs": spec.output_names,
            "dynamic_axes": spec.dynamic_axes,
            "input_dtypes": spec.input_dtypes,
            "output_dtypes": spec.output_dtypes,
            "required_for_runtime": spec.required_for_runtime,
        }

    manifest = {
        "schema_version": 2,
        "model": "chatterbox-vc",
        "checkpoint": {
            "path": str(config.checkpoint_dir),
            "hash": checkpoint_hash(config.checkpoint_dir),
        },
        "onnx": {
            "opset": config.opset,
            "external_data": config.external_data,
            "precisions": list(config.precisions),
        },
        "constants": _constants(source_hop),
        "graphs": graphs,
        "artifacts": [asdict(artifact) for artifact in artifacts],
    }
    if ort_version is not None:
        manifest["onnxruntime_version"] = ort_version

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    metadata = {
        k: manifest[k]
        for k in ("schema_version", "model", "checkpoint", "onnx", "constants")
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


def load_manifest(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing ONNX manifest: {path}")
    return json.loads(path.read_text())
