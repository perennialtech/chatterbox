from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactRecord
from .errors import OnnxExportError


class ExportSession:
    def __init__(self, opset: int = 18, external_data: bool = True):
        self.opset = opset
        self.external_data = external_data

    def export(
        self,
        graph_name: str,
        module: torch.nn.Module,
        path: Path,
        inputs: tuple[torch.Tensor, ...],
        input_names: list[str],
        output_names: list[str],
        dynamic_shapes: tuple[Any, ...] | dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        path.parent.mkdir(parents=True, exist_ok=True)
        module.eval()

        for m in module.modules():
            if isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                assert not m.training, f"BatchNorm still training: {m}"

        with torch.inference_mode():
            export_kwargs = {
                "input_names": input_names,
                "output_names": output_names,
                "opset_version": self.opset,
                "do_constant_folding": True,
                "external_data": self.external_data,
                "dynamo": True,
                "optimize": True,
            }

            if dynamic_shapes:
                if isinstance(dynamic_shapes, dict):
                    filtered_dynamic_shapes = {
                        k: v for k, v in dynamic_shapes.items() if k in input_names
                    }
                    if filtered_dynamic_shapes:
                        export_kwargs["dynamic_shapes"] = filtered_dynamic_shapes
                else:
                    export_kwargs["dynamic_shapes"] = dynamic_shapes

            export_start = time.perf_counter()
            torch.onnx.export(module, inputs, str(path), **export_kwargs)
            export_seconds = time.perf_counter() - export_start

        check_start = time.perf_counter()
        self.check(path)
        check_seconds = time.perf_counter() - check_start

        size_mib = _artifact_size_bytes(path) / (1024 * 1024)
        print(
            f"Exported {graph_name}: export={export_seconds:.2f}s "
            f"check={check_seconds:.2f}s size={size_mib:.2f} MiB"
        )

        return ArtifactRecord(
            graph_name=graph_name,
            path=str(path),
            inputs=input_names,
            outputs=output_names,
            dynamic_shapes=dynamic_shapes or {},
        )

    @staticmethod
    def check(path: Path) -> None:
        try:
            import onnx
        except ImportError as exc:
            raise OnnxExportError("onnx is required to check exported graphs") from exc

        onnx.checker.check_model(str(path))


def _artifact_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for candidate in path.parent.glob(f"{path.name}*"):
        if candidate.is_file():
            total += candidate.stat().st_size
    return total
