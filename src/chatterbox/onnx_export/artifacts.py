from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from ..audio import (DEC_COND_LEN, ENC_COND_LEN, MEL_HOP_24K, S3_HOP, S3_SR,
                     S3_TOKEN_HOP, S3_TOKEN_RATE, S3GEN_SR, SPEECH_VOCAB_SIZE)
from ..models.s3gen.const import S3GEN_SIL
from ..models.s3gen.pipeline import (_TOKEN_LENGTH_BUCKETS, FLOW_CHUNK_TOKENS,
                                     FLOW_CONTEXT_TOKENS,
                                     REF_MAX_PROMPT_TOKENS, REF_MAX_SECONDS,
                                     REF_MIN_PROMPT_TOKENS, REF_MIN_SECONDS)
from .buckets import (FLOW_MEL_BUCKETS, TOKEN_TO_MU_TOKEN_BUCKETS,
                      VOCODER_MEL_BUCKETS)
from .config import ExportConfig
from .constants import MEANFLOW_T_SPAN
from .graph_spec import GraphSpec


@dataclass
class ArtifactRecord:
    graph_name: str
    path: str
    inputs: list[str]
    outputs: list[str]
    dynamic_shapes: Any


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def checkpoint_hash(checkpoint_dir: Path) -> str:
    checkpoint_path = checkpoint_dir.resolve() / "s3gen_meanflow.safetensors"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint file: {checkpoint_path}")
    return sha256_file(checkpoint_path)


def manifest_hash(path: Path) -> str:
    return sha256_file(path)


def _constants(
    *,
    source_hop: int,
    token_mel_ratio: int,
    final_context_token_count: int,
    vocoder_harmonics: int,
) -> dict[str, Any]:
    n_trim = S3GEN_SR // 50
    return {
        "sample_rates": {"s3": S3_SR, "s3gen": S3GEN_SR},
        "hop_sizes": {"s3": S3_HOP, "s3_token": S3_TOKEN_HOP, "mel_24k": MEL_HOP_24K},
        "token_rate": S3_TOKEN_RATE,
        "speech_vocab_size": SPEECH_VOCAB_SIZE,
        "s3gen_sil": S3GEN_SIL,
        "source_hop": source_hop,
        "meanflow_t_span": list(MEANFLOW_T_SPAN),
        "prompt_limits": {
            "encoder_samples": ENC_COND_LEN,
            "decoder_samples": DEC_COND_LEN,
            "ref_min_seconds": REF_MIN_SECONDS,
            "ref_max_seconds": REF_MAX_SECONDS,
            "ref_min_prompt_tokens": REF_MIN_PROMPT_TOKENS,
            "ref_max_prompt_tokens": REF_MAX_PROMPT_TOKENS,
        },
        "flow_chunk_tokens": FLOW_CHUNK_TOKENS,
        "flow_context_tokens": FLOW_CONTEXT_TOKENS,
        "token_length_buckets": list(_TOKEN_LENGTH_BUCKETS),
        "token_to_mu_token_buckets": list(TOKEN_TO_MU_TOKEN_BUCKETS),
        "flow_mel_buckets": list(FLOW_MEL_BUCKETS),
        "vocoder_mel_buckets": list(VOCODER_MEL_BUCKETS),
        "token_mel_ratio": token_mel_ratio,
        "final_context_token_count": final_context_token_count,
        "vocoder_harmonics": vocoder_harmonics,
        "fade": {
            "n_trim": n_trim,
            "trim_fade_len": 2 * n_trim,
            "end_fade_len": 2 * n_trim,
        },
    }


def write_manifest(
    output_dir: Path,
    config: ExportConfig,
    graph_specs: tuple[GraphSpec, ...],
    artifacts: list[ArtifactRecord],
    source_hop: int,
    token_mel_ratio: int,
    final_context_token_count: int,
    vocoder_harmonics: int,
    ort_version: str | None = None,
) -> None:
    output_dir = output_dir.resolve()
    checkpoint_dir = config.checkpoint_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    artifact_map = {artifact.graph_name: artifact for artifact in artifacts}

    graphs: dict[str, Any] = {}
    for spec in graph_specs:
        artifact = artifact_map.get(spec.name)
        if artifact is None:
            raise RuntimeError(f"Missing ONNX artifact for graph {spec.name}")

        artifact_path = Path(artifact.path).resolve()
        graphs[spec.name] = {
            "filename": spec.filename,
            "path": str(artifact_path.relative_to(output_dir)),
            "inputs": spec.input_names,
            "outputs": spec.output_names,
            "dynamic_shapes": _serialize_dynamic_shapes(spec.dynamic_shapes),
            "input_dtypes": spec.input_dtypes,
            "output_dtypes": spec.output_dtypes,
            "required_for_runtime": spec.required_for_runtime,
        }

    serializable_artifacts = []
    for artifact in artifacts:
        art_dict = asdict(artifact)
        art_dict["path"] = str(Path(art_dict["path"]).resolve().relative_to(output_dir))
        art_dict["dynamic_shapes"] = _serialize_dynamic_shapes(
            art_dict["dynamic_shapes"]
        )
        serializable_artifacts.append(art_dict)

    manifest = {
        "schema_version": 4,
        "model": "chatterbox-vc",
        "checkpoint": {
            "path": str(checkpoint_dir),
            "hash": checkpoint_hash(checkpoint_dir),
            "hash_file": "s3gen_meanflow.safetensors",
        },
        "onnx": {
            "opset": config.opset,
            "external_data": config.external_data,
            "batch_size": 1,
            "bucket_strategy": "static",
            "exporter_optimization": True,
        },
        "constants": _constants(
            source_hop=source_hop,
            token_mel_ratio=token_mel_ratio,
            final_context_token_count=final_context_token_count,
            vocoder_harmonics=vocoder_harmonics,
        ),
        "graphs": graphs,
        "artifacts": serializable_artifacts,
    }
    if ort_version is not None:
        manifest["onnxruntime_version"] = ort_version

    metadata = {
        k: manifest[k]
        for k in ("schema_version", "model", "checkpoint", "onnx", "constants")
    }

    manifest_text = json.dumps(manifest, indent=2, sort_keys=True)
    metadata_text = json.dumps(metadata, indent=2, sort_keys=True)

    manifest_tmp = output_dir / "manifest.json.tmp"
    metadata_tmp = output_dir / "metadata.json.tmp"
    manifest_tmp.write_text(manifest_text)
    metadata_tmp.write_text(metadata_text)
    manifest_tmp.replace(output_dir / "manifest.json")
    metadata_tmp.replace(output_dir / "metadata.json")


def load_manifest(artifact_dir: Path) -> dict[str, Any]:
    path = Path(artifact_dir).resolve() / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing ONNX manifest: {path}")
    return json.loads(path.read_text())


def _serialize_dynamic_shapes(shapes: Any) -> Any:
    if isinstance(shapes, dict):
        return {str(k): _serialize_dynamic_shapes(v) for k, v in shapes.items()}
    elif isinstance(shapes, (list, tuple)):
        return [_serialize_dynamic_shapes(v) for v in shapes]
    elif str(type(shapes)).find("Dim") != -1 or hasattr(shapes, "__name__"):
        return getattr(shapes, "__name__", str(shapes))
    return shapes
