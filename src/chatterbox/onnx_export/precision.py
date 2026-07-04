from __future__ import annotations

from pathlib import Path

from .errors import OnnxExportError


def convert_fp16(src: Path, dst: Path) -> None:
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        raise OnnxExportError(
            "onnx and onnxconverter-common are required for FP16 conversion"
        ) from exc

    model = onnx.load(str(src))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(dst))
    checked = onnx.load(str(dst), load_external_data=False)
    onnx.checker.check_model(checked)
