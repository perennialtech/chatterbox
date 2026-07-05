from __future__ import annotations

import re

from .errors import TensorRTError


def tensorrt_major_version(trt_module) -> int:
    version = str(getattr(trt_module, "__version__", ""))
    match = re.match(r"^\s*(\d+)", version)
    if match is None:
        raise TensorRTError(f"Unable to parse TensorRT version: {version!r}")
    return int(match.group(1))


def require_tensorrt_10(
    trt_module, error_type: type[TensorRTError] = TensorRTError
) -> None:
    major = tensorrt_major_version(trt_module)
    if major < 10:
        version = str(getattr(trt_module, "__version__", "unknown"))
        raise error_type(f"TensorRT >= 10 is required; found TensorRT {version}")


def network_creation_flags(trt_module, *, strongly_typed: bool) -> int:
    flags = 0
    if strongly_typed and hasattr(
        trt_module.NetworkDefinitionCreationFlag, "STRONGLY_TYPED"
    ):
        flags |= 1 << int(trt_module.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
    return flags
