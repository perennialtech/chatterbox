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
    "reference_mel_24k": Tolerance(1e-3, 2e-4),
    "speaker_encoder": CosineTolerance(0.999),
    "token_to_mu": Tolerance(1e-3, 2e-4),
    "conditional_decoder_step": Tolerance(3e-2, 3e-3),
    "flow_decoder_meanflow2": Tolerance(4e-2, 4e-3),
    "vocoder_hift": Tolerance(2e-2, 5e-3),
}
