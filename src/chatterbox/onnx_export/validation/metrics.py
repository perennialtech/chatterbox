from __future__ import annotations

import numpy as np
import torch

from ..errors import OnnxValidationError
from .tolerances import CosineTolerance, Tolerance


def to_numpy(value) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def compare_exact(name: str, expected, actual) -> dict:
    expected_np = to_numpy(expected)
    actual_np = np.asarray(actual)
    if expected_np.shape != actual_np.shape or not np.array_equal(
        expected_np, actual_np
    ):
        raise OnnxValidationError(
            f"{name} exact parity failed: expected {expected_np.shape}, actual {actual_np.shape}"
        )
    return {"exact": True}


def compare_tensors(name: str, expected, actual, tolerance: Tolerance) -> dict:
    expected_np = to_numpy(expected).astype(np.float32)
    actual_np = np.asarray(actual).astype(np.float32)
    if expected_np.shape != actual_np.shape:
        raise OnnxValidationError(
            f"{name} shape mismatch: expected {expected_np.shape}, actual {actual_np.shape}"
        )
    diff = np.abs(expected_np - actual_np)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    if max_abs > tolerance.max_abs or mean_abs > tolerance.mean_abs:
        raise OnnxValidationError(
            f"{name} parity failed: max_abs={max_abs}, mean_abs={mean_abs}, tolerance={tolerance}"
        )
    return {"max_abs": max_abs, "mean_abs": mean_abs}


def compare_cosine(name: str, expected, actual, tolerance: CosineTolerance) -> dict:
    expected_np = to_numpy(expected).astype(np.float32).reshape(expected.shape[0], -1)
    actual_np = np.asarray(actual).astype(np.float32).reshape(actual.shape[0], -1)
    numerator = np.sum(expected_np * actual_np, axis=1)
    denom = np.linalg.norm(expected_np, axis=1) * np.linalg.norm(actual_np, axis=1)
    cosine = numerator / np.maximum(denom, 1e-12)
    min_cos = float(cosine.min()) if cosine.size else 1.0
    if min_cos < tolerance.min_cosine:
        raise OnnxValidationError(
            f"{name} cosine parity failed: min_cosine={min_cos}, tolerance={tolerance}"
        )
    return {"min_cosine": min_cos}
