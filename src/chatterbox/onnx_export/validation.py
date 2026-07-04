from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .errors import OnnxValidationError


@dataclass(frozen=True)
class Tolerance:
    max_abs: float
    mean_abs: float


DEFAULT_TOLERANCES = {
    "token_to_mu_fp32": Tolerance(1e-4, 1e-5),
    "conditional_decoder_step_fp32": Tolerance(2e-3, 2e-4),
    "conditional_decoder_step_fp16": Tolerance(2e-2, 2e-3),
    "vocoder_fp32": Tolerance(1e-4, 1e-4),
    "vocoder_fp16": Tolerance(5e-3, 5e-3),
}


def compare_tensors(
    name: str, torch_out: torch.Tensor, ort_out: np.ndarray, tolerance: Tolerance
) -> dict:
    expected = torch_out.detach().cpu().float().numpy()
    actual = ort_out.astype(np.float32)
    diff = np.abs(expected - actual)
    max_abs = float(diff.max()) if diff.size else 0.0
    mean_abs = float(diff.mean()) if diff.size else 0.0
    if max_abs > tolerance.max_abs or mean_abs > tolerance.mean_abs:
        raise OnnxValidationError(
            f"{name} parity failed: max_abs={max_abs}, mean_abs={mean_abs}, tolerance={tolerance}"
        )
    return {"max_abs": max_abs, "mean_abs": mean_abs}


def run_ort(path: Path, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise OnnxValidationError("onnxruntime is required for validation") from exc
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, inputs)
