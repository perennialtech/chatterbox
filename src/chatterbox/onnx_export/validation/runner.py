from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ..artifacts import load_manifest
from ..config import SinglePrecision
from ..constants import GRAPH_S3_TOKENIZER_QUANTIZER, GRAPH_SPEAKER_ENCODER
from ..graphs import ALL_GRAPHS
from ..model_loading import load_torch_model
from .metrics import compare_cosine, compare_exact, compare_tensors
from .tolerances import DEFAULT_TOLERANCES, CosineTolerance, Tolerance


def _ort_providers(ort, device: str) -> list[str]:
    available = set(ort.get_available_providers())
    if str(device).startswith("cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _run_ort(
    path: Path,
    input_names: list[str],
    inputs: tuple[torch.Tensor, ...],
    device: str = "cpu",
) -> list[np.ndarray]:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=_ort_providers(ort, device))
    actual_input_names = {inp.name for inp in session.get_inputs()}

    feed = {
        name: np.ascontiguousarray(t.detach().cpu().numpy())
        for name, t in zip(input_names, inputs)
        if name in actual_input_names
    }

    return session.run(None, feed)


def run_validation(
    artifact_dir: Path,
    checkpoint_dir: Path,
    precision: SinglePrecision,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(1234)
    rng = np.random.default_rng(1234)
    _ = rng.random()

    manifest = load_manifest(artifact_dir)
    model = load_torch_model(Path(checkpoint_dir), device=device)
    report: dict[str, dict] = {"precision": precision, "graphs": {}}

    for spec in ALL_GRAPHS:
        graph_entry = manifest["graphs"][spec.name]
        onnx_path = artifact_dir / graph_entry["files"][precision]
        module = spec.make_module(model).to(device).eval()

        if spec.name == "vocoder_hift":
            inputs = spec.make_dummy_inputs()
            source_hop = int(manifest["constants"]["source_hop"])
            inputs = (
                inputs[0],
                inputs[1],
                torch.randn(
                    inputs[0].size(0),
                    9,
                    inputs[0].size(2) * source_hop,
                    dtype=torch.float32,
                ),
            )
        else:
            inputs = spec.make_dummy_inputs()

        inputs = tuple(x.to(device) for x in inputs)

        with torch.inference_mode():
            torch_outputs = module(*inputs)
        if not isinstance(torch_outputs, (tuple, list)):
            torch_outputs = (torch_outputs,)

        ort_outputs = _run_ort(onnx_path, spec.input_names, inputs, device=device)
        graph_report = {}

        for output_name, expected, actual in zip(
            spec.output_names, torch_outputs, ort_outputs
        ):
            key = f"{spec.name}_{precision}"
            if not torch.is_floating_point(expected):
                graph_report[output_name] = compare_exact(
                    f"{spec.name}.{output_name}", expected, actual
                )
            elif spec.name == GRAPH_S3_TOKENIZER_QUANTIZER:
                graph_report[output_name] = compare_exact(
                    f"{spec.name}.{output_name}", expected, actual
                )
            elif spec.name == GRAPH_SPEAKER_ENCODER:
                tol = DEFAULT_TOLERANCES[key]
                assert isinstance(tol, CosineTolerance)
                graph_report[output_name] = compare_cosine(
                    f"{spec.name}.{output_name}", expected, actual, tol
                )
            else:
                tol = DEFAULT_TOLERANCES[key]
                assert isinstance(tol, Tolerance)
                graph_report[output_name] = compare_tensors(
                    f"{spec.name}.{output_name}", expected, actual, tol
                )

        report["graphs"][spec.name] = graph_report

    out_dir = artifact_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{precision}.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    return report


def run_validation_for_precisions(
    artifact_dir: Path,
    checkpoint_dir: Path,
    precisions: tuple[SinglePrecision, ...],
    device: str = "cpu",
) -> dict[str, dict]:
    return {
        precision: run_validation(
            artifact_dir, checkpoint_dir, precision, device=device
        )
        for precision in precisions
    }
