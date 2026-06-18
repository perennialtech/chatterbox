from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MelGeneratorConfig:
    vocab_size: int = 6561
    token_dim: int = 512
    mel_bins: int = 80
    speaker_embedding_dim: int = 192
    projected_speaker_dim: int = 80
    pre_lookahead_tokens: int = 3
    default_meanflow_steps: int = 2
