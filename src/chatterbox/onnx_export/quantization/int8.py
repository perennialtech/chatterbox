from pathlib import Path


def quantize_int8_dynamic(src: Path, dst: Path) -> None:
    from onnxruntime.quantization import QuantType, quantize_dynamic

    dst.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
