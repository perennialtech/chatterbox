from pathlib import Path


def convert_fp16(src: Path, dst: Path) -> None:
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    model_fp16 = float16.convert_float_to_float16(model, keep_io_types=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model_fp16, str(dst))
