from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    max_abs: float
    mean_abs: float


@dataclass(frozen=True)
class CosineTolerance:
    min_cosine: float


DEFAULT_TOLERANCES = {
    "s3_tokenizer_log_mel": Tolerance(3e-3, 5e-6),
    "reference_mel_24k": Tolerance(1e-3, 2e-4),
    "speaker_encoder": CosineTolerance(0.999),
    "token_to_mu": Tolerance(1e-3, 2e-4),
    "flow_decoder_meanflow2": Tolerance(4e-2, 4e-3),
    "vocoder_hift": Tolerance(3e-2, 2e-3),
    "full_pipeline_mel": Tolerance(4e-2, 4e-3),
    "full_pipeline_wav": Tolerance(3e-2, 2e-3),
}


def tolerance_for_graph(graph_name: str):
    for key, tolerance in DEFAULT_TOLERANCES.items():
        if graph_name == key or graph_name.startswith(f"{key}_"):
            return tolerance
    raise KeyError(f"No validation tolerance configured for graph {graph_name}")
