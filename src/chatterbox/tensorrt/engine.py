from __future__ import annotations

from pathlib import Path

import numpy as np

from .cuda import (CudaStream, DeviceBuffer, memcpy_dtoh_async,
                   memcpy_htod_async)
from .errors import TensorRTRuntimeError, TensorRTShapeError


class TrtEngineRunner:
    def __init__(self, engine_path: Path, logger_level=None):
        import tensorrt as trt

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
        self.stream = CudaStream()
        self.buffers: dict[str, DeviceBuffer] = {}

        self._input_names = []
        self._output_names = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            else:
                self._output_names.append(name)

    @property
    def input_names(self) -> list[str]:
        return list(self._input_names)

    @property
    def output_names(self) -> list[str]:
        return list(self._output_names)

    def _profile_range(self, name: str):
        try:
            return self.engine.get_tensor_profile_shape(name, 0)
        except Exception:
            return None

    def _validate_shape(self, name: str, shape: tuple[int, ...]) -> None:
        profile = self._profile_range(name)
        if profile is None:
            return
        mn, _, mx = profile
        if len(shape) != len(mn):
            raise TensorRTShapeError(
                f"{self.engine_path.name}:{name} rank {len(shape)} not in profile rank {len(mn)}"
            )
        for actual, lo, hi in zip(shape, mn, mx):
            if actual < lo or actual > hi:
                raise TensorRTShapeError(
                    f"{self.engine_path.name}:{name} shape {shape} outside profile min={tuple(mn)} max={tuple(mx)}"
                )

    def _buffer(self, name: str, nbytes: int) -> int:
        buf = self.buffers.setdefault(name, DeviceBuffer())
        return buf.ensure_capacity(max(1, int(nbytes)))

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        for name in self._input_names:
            if name not in inputs:
                raise TensorRTRuntimeError(
                    f"{self.engine_path.name}: missing input {name}"
                )
            arr = np.ascontiguousarray(inputs[name])
            expected_dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            if arr.dtype != expected_dtype:
                arr = arr.astype(expected_dtype)
            self._validate_shape(name, tuple(arr.shape))
            if not self.context.set_input_shape(name, tuple(arr.shape)):
                raise TensorRTShapeError(
                    f"{self.engine_path.name}: failed to set input shape for {name}: {arr.shape}"
                )
            inputs[name] = arr

        for name in self._input_names:
            arr = inputs[name]
            ptr = self._buffer(name, arr.nbytes)
            memcpy_htod_async(ptr, arr, self.stream)
            self.context.set_tensor_address(name, ptr)

        outputs: dict[str, np.ndarray] = {}
        for name in self._output_names:
            shape = tuple(int(dim) for dim in self.context.get_tensor_shape(name))
            if any(dim < 0 for dim in shape):
                raise TensorRTRuntimeError(
                    f"{self.engine_path.name}: unresolved output shape for {name}: {shape}"
                )
            dtype = self.trt.nptype(self.engine.get_tensor_dtype(name))
            arr = np.empty(shape, dtype=dtype)
            ptr = self._buffer(name, arr.nbytes)
            self.context.set_tensor_address(name, ptr)
            outputs[name] = arr

        if not self.context.execute_async_v3(stream_handle=self.stream.handle):
            raise TensorRTRuntimeError(
                f"{self.engine_path.name}: execute_async_v3 failed"
            )

        for name, arr in outputs.items():
            ptr = self.buffers[name].ptr
            assert ptr is not None
            memcpy_dtoh_async(arr, ptr, self.stream)
        self.stream.synchronize()
        return outputs

    def close(self) -> None:
        for buffer in self.buffers.values():
            buffer.free()
        self.buffers.clear()
        self.stream.close()
