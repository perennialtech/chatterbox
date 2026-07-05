from __future__ import annotations

import numpy as np


def trim_fade(length: int, dtype=np.float32) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=dtype)

    half = length // 2
    fade = np.zeros(length, dtype=dtype)
    fade[half:] = (np.cos(np.linspace(np.pi, 0, length - half)) + 1) / 2
    return fade


def apply_initial_trim_fade(wav: np.ndarray, length: int) -> np.ndarray:
    if length <= 0 or wav.shape[-1] == 0:
        return wav

    n = min(int(length), int(wav.shape[-1]))
    wav[..., :n] *= trim_fade(length, wav.dtype)[:n]
    return wav
