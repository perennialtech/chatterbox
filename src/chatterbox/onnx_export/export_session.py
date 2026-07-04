from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .artifacts import ArtifactRecord
from .config import SinglePrecision
from .errors import OnnxExportError


class ExportSession:
    def __init__(self, opset: int = 18, external_data: bool = True):
        self.opset = opset
        self.external_data = external_data

    def export(
        self,
        graph_name: str,
        precision: SinglePrecision,
        module: torch.nn.Module,
        path: Path,
        inputs: tuple[torch.Tensor, ...],
        input_names: list[str],
        output_names: list[str],
        dynamic_axes: dict[str, Any] | None = None,
    ) -> ArtifactRecord:
        path.parent.mkdir(parents=True, exist_ok=True)
        module.eval()

        with torch.inference_mode():
            try:
                torch.onnx.export(
                    module,
                    inputs,
                    str(path),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=self.opset,
                    do_constant_folding=True,
                    external_data=self.external_data,
                    dynamo=True,
                )
            except TypeError:
                torch.onnx.export(
                    module,
                    inputs,
                    str(path),
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    opset_version=self.opset,
                    do_constant_folding=True,
                )
            except Exception as exc:
                raise OnnxExportError(f"Failed to export {path.name}: {exc}") from exc

        self.check(path)
        return ArtifactRecord(
            graph_name=graph_name,
            precision=precision,
            path=str(path),
            inputs=input_names,
            outputs=output_names,
            dynamic_axes=dynamic_axes or {},
        )

    @staticmethod
    def check(path: Path) -> None:
        try:
            import onnx
        except ImportError as exc:
            raise OnnxExportError("onnx is required to check exported graphs") from exc
        model = onnx.load(str(path), load_external_data=False)
        onnx.checker.check_model(model)
