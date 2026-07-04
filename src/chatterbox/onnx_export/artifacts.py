import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..audio import (DEC_COND_LEN, ENC_COND_LEN, MEL_HOP_24K, S3_HOP, S3_SR,
                     S3_TOKEN_HOP, S3_TOKEN_RATE, S3GEN_SR, SPEECH_VOCAB_SIZE)
from .config import ExportConfig


@dataclass
class ArtifactRecord:
    name: str
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


def base_metadata(config: ExportConfig) -> dict[str, Any]:
    return {
        "model": "chatterbox-vc",
        "checkpoint_hash": checkpoint_hash(config.checkpoint_dir),
        "opset": config.opset,
        "profile": config.profile,
        "precision": config.precision,
        "quantization": config.quantization,
        "external_data": config.external_data,
        "sample_rates": {"s3": S3_SR, "s3gen": S3GEN_SR},
        "hop_sizes": {"s3": S3_HOP, "s3_token": S3_TOKEN_HOP, "mel_24k": MEL_HOP_24K},
        "token_rate": S3_TOKEN_RATE,
        "speech_vocab_size": SPEECH_VOCAB_SIZE,
        "prompt_limits": {
            "encoder_samples": ENC_COND_LEN,
            "decoder_samples": DEC_COND_LEN,
        },
        "buckets": list(config.buckets),
    }


def write_manifest(
    output_dir: Path,
    config: ExportConfig,
    artifacts: list[ArtifactRecord],
    ort_version: str | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = base_metadata(config)
    if ort_version is not None:
        metadata["onnxruntime_version"] = ort_version

    manifest = {
        "metadata": metadata,
        "artifacts": [asdict(artifact) for artifact in artifacts],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
