from __future__ import annotations

from pathlib import Path

import numpy as np

from .api import require_tensorrt_10
from .cuda import (CudaStream, DeviceBuffer, memcpy_dtoh_async,
                   memcpy_htod_async)
from .errors import TensorRTRuntimeError, TensorRTShapeError


class TrtEngineRunner:
    def __init__(self, engine_path: Path, logger_level=None):
        import tensorrt as trt

        require_tensorrt_10(trt, TensorRTRuntimeError)

        self.trt = trt
        self.engine_path = Path(engine_path)
        if logger_level is None:
            logger_level = trt.Logger.WARNING

        self.logger = trt.Logger(logger_level)
        self.runtime = trt.Runtime(self.logger)
        serialized = self.engine_path.read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(serialized)
        if self.engine is None:
            raise TensorRTRuntimeError(
                f"Failed to deserialize TensorRT engine: {engine_path}"
            )

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise TensorRTRuntimeError(
                f"Failed to create TensorRT execution context: {engine_path}"
            )

        self.stream: CudaStream | None = CudaStream()
        self.buffers: dict[str, DeviceBuffer] = {}

        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self._output_names.append(name)
            else:
                raise TensorRTRuntimeError(
                    f"{self.engine_path.name}: unsupported TensorRT tensor mode for {name}: {mode}"
                )

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    def _profile_range(
        self, name: str
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]] | None:
        try:
            profile = self.engine.get_tensor_profile_shape(name, 0)
        except Exception:
            return None

        if profile is None or len(profile) != 3:
            return None

        mn, opt, mx = profile
        return (
            tuple(int(dim) for dim in mn),
            tuple(int(dim) for dim in opt),
            tuple(int(dim) for dim in mx),
        )

    def _declared_shape(self, name: str) -> tuple[int, ...]:
        return tuple(int(dim) for dim in self.engine.get_tensor_shape(name))

    def _validate_shape(self, name: str, shape: tuple[int, ...]) -> None:
        profile = self._profile_range(name)
        if profile is not None:
            mn, _, mx = profile
            if len(shape) != len(mn):
                raise TensorRTShapeError(
                    f"{self.engine_path.name}:{name} rank {len(shape)} does not match profile rank {len(mn)}"
                )
            for actual, lo, hi in zip(shape, mn, mx):
                if actual < lo or actual > hi:
                    raise TensorRTShapeError(
                        f"{self.engine_path.name}:{name} shape {shape} outside profile min={mn} max={mx}"
                    )
            return

        declared = self._declared_shape(name)
        if len(shape) != len(declared):
            raise TensorRTShapeError(
                f"{self.engine_path.name}:{name} rank {len(shape)} does not match engine rank {len(declared)}"
            )

        for actual, expected in zip(shape, declared):
            if expected >= 0 and actual != expected:
                raise TensorRTShapeError(
                    f"{self.engine_path.name}:{name} shape {shape} does not match engine shape {declared}"
                )

    def _stream(self) -> CudaStream:
        if self.stream is None:
            raise TensorRTRuntimeError(f"{self.engine_path.name}: runner is closed")
        return self.stream

    def _buffer(self, name: str, nbytes: int) -> int:
        buf = self.buffers.setdefault(name, DeviceBuffer())
        return buf.ensure_capacity(max(1, int(nbytes)))

    def _set_tensor_address(self, name: str, ptr: int) -> None:
        if not self.context.set_tensor_address(name, ptr):
            raise TensorRTRuntimeError(
                f"{self.engine_path.name}: failed to set tensor address for {name}"
            )

    def _infer_shapes(self) -> None:
        missing = self.context.infer_shapes()
        if missing:
            names = ", ".join(str(name) for name in missing)
            raise TensorRTShapeError(
                f"{self.engine_path.name}: insufficient shape information for {names}"
            )

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        stream = self._stream()
        prepared_inputs: dict[str, np.ndarray] = {}

        for name in self._input_names:
            if name not in inputs:
                raise TensorRTRuntimeError(
                    f"{self.engine_path.name}: missing input {name}"
                )

            expected_dtype = np.dtype(
                self.trt.nptype(self.engine.get_tensor_dtype(name))
            )
            arr = np.ascontiguousarray(inputs[name])
            if arr.dtype != expected_dtype:
                arr = arr.astype(expected_dtype)

            shape = tuple(int(dim) for dim in arr.shape)
            self._validate_shape(name, shape)
            if not self.context.set_input_shape(name, shape):
                raise TensorRTShapeError(
                    f"{self.engine_path.name}: failed to set input shape for {name}: {shape}"
                )

            prepared_inputs[name] = arr

        for name in self._input_names:
            arr = prepared_inputs[name]
            ptr = self._buffer(name, arr.nbytes)
            memcpy_htod_async(ptr, arr, stream)
            self._set_tensor_address(name, ptr)

        self._infer_shapes()

        outputs: dict[str, np.ndarray] = {}
        for name in self._output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise TensorRTRuntimeError(
                    f"{self.engine_path.name}: unresolved output shape for {name}: {shape}"
                )

            dtype = np.dtype(self.trt.nptype(self.engine.get_tensor_dtype(name)))
            arr = np.empty(shape, dtype=dtype)
            ptr = self._buffer(name, arr.nbytes)
            self._set_tensor_address(name, ptr)
            outputs[name] = arr

        if not self.context.execute_async_v3(stream_handle=stream.handle):
            raise TensorRTRuntimeError(
                f"{self.engine_path.name}: execute_async_v3 failed"
            )

        for name, arr in outputs.items():
            ptr = self.buffers[name].ptr
            assert ptr is not None
            memcpy_dtoh_async(arr, ptr, stream)

        stream.synchronize()
        return outputs

    def close(self) -> None:
        for buffer in self.buffers.values():
            buffer.free()
        self.buffers.clear()

        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def __enter__(self) -> TrtEngineRunner:
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
