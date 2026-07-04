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
    "reference_mel_24k_fp32": Tolerance(1e-4, 1e-5),
    "reference_mel_24k_fp16": Tolerance(5e-2, 5e-3),
    "speaker_encoder_fp32": CosineTolerance(0.999),
    "speaker_encoder_fp16": CosineTolerance(0.995),
    "token_to_mu_fp32": Tolerance(1e-4, 1e-5),
    "token_to_mu_fp16": Tolerance(2e-2, 2e-3),
    "conditional_decoder_step_fp32": Tolerance(2e-3, 2e-4),
    "conditional_decoder_step_fp16": Tolerance(2e-2, 2e-3),
    "flow_decoder_meanflow2_fp32": Tolerance(4e-3, 4e-4),
    "flow_decoder_meanflow2_fp16": Tolerance(4e-2, 4e-3),
    "vocoder_hift_fp32": Tolerance(1e-4, 1e-4),
    "vocoder_hift_fp16": Tolerance(5e-3, 5e-3),
}
