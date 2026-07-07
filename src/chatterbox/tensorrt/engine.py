from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from .api import require_tensorrt_10
from .cuda import (CudaStream, DeviceBuffer, memcpy_dtoh_async,
                   memcpy_htod_async)
from .errors import TensorRTRuntimeError, TensorRTShapeError
from .memory import TrtActivationMemoryPool
from .runtime_api import (create_user_managed_context,
                          engine_device_memory_size_v2,
                          set_context_device_memory)


class TrtEngineRunner:
    def __init__(
        self,
        engine_path: Path,
        logger_level=None,
        activation_pool: TrtActivationMemoryPool | None = None,
    ):
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

        self.activation_memory_nbytes = engine_device_memory_size_v2(self.engine)
        self.activation_pool = activation_pool
        self._run_lock = threading.RLock()

        if self.activation_pool is not None:
            if self.activation_pool.nbytes < self.activation_memory_nbytes:
                raise TensorRTRuntimeError(
                    f"{self.engine_path.name}: shared TensorRT activation pool is too small; "
                    f"need {self.activation_memory_nbytes} bytes, got {self.activation_pool.nbytes}"
                )
            if self.activation_pool.ptr is None:
                raise TensorRTRuntimeError(
                    f"{self.engine_path.name}: shared TensorRT activation pool is closed"
                )
            self.context = create_user_managed_context(trt, self.engine)
        else:
            self.context = self.engine.create_execution_context()

        if self.context is None:
            raise TensorRTRuntimeError(
                f"Failed to create TensorRT execution context: {engine_path}"
            )

        if self.activation_pool is not None:
            assert self.activation_pool.ptr is not None
            set_context_device_memory(
                self.context, self.activation_pool.ptr, self.activation_pool.nbytes
            )

        self.stream: CudaStream | None = CudaStream()
        self.buffers: dict[str, DeviceBuffer] = {}

        self._input_names: list[str] = []
        self._output_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self._validate_linear_io_tensor(name)
            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self._input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self._output_names.append(name)
            else:
                raise TensorRTRuntimeError(
                    f"{self.engine_path.name}: unsupported TensorRT tensor mode for {name}: {mode}"
                )

    def _validate_linear_io_tensor(self, name: str) -> None:
        get_format = getattr(self.engine, "get_tensor_format", None)
        tensor_format = getattr(self.trt, "TensorFormat", None)
        linear_format = (
            getattr(tensor_format, "LINEAR", None) if tensor_format else None
        )

        if get_format is not None and linear_format is not None:
            actual_format = get_format(name)
            if actual_format != linear_format:
                actual_name = getattr(actual_format, "name", str(actual_format))
                raise TensorRTRuntimeError(
                    f"{self.engine_path.name}:{name} uses TensorRT I/O format "
                    f"{actual_name}; rebuild the engine with linear I/O formats"
                )

        get_vectorized_dim = getattr(self.engine, "get_tensor_vectorized_dim", None)
        if get_vectorized_dim is not None:
            vectorized_dim = int(get_vectorized_dim(name))
            if vectorized_dim != -1:
                raise TensorRTRuntimeError(
                    f"{self.engine_path.name}:{name} uses vectorized TensorRT I/O "
                    f"dimension {vectorized_dim}; rebuild the engine with linear I/O formats"
                )

    @staticmethod
    def inspect_activation_memory(engine_path: Path, logger_level=None) -> int:
        import tensorrt as trt

        require_tensorrt_10(trt, TensorRTRuntimeError)

        if logger_level is None:
            logger_level = trt.Logger.WARNING

        logger = trt.Logger(logger_level)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        if engine is None:
            raise TensorRTRuntimeError(
                f"Failed to deserialize TensorRT engine: {engine_path}"
            )

        try:
            return engine_device_memory_size_v2(engine)
        finally:
            engine = None
            runtime = None

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
        if self.context is None:
            raise TensorRTRuntimeError(f"{self.engine_path.name}: runner is closed")
        if not self.context.set_tensor_address(name, ptr):
            raise TensorRTRuntimeError(
                f"{self.engine_path.name}: failed to set tensor address for {name}"
            )

    def _infer_shapes(self) -> None:
        if self.context is None:
            raise TensorRTRuntimeError(f"{self.engine_path.name}: runner is closed")
        missing = self.context.infer_shapes()
        if missing:
            names = ", ".join(str(name) for name in missing)
            raise TensorRTShapeError(
                f"{self.engine_path.name}: insufficient shape information for {names}"
            )

    def run(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        with self._run_lock:
            if self.activation_pool is None:
                return self._run_impl(inputs)

            with self.activation_pool.lock:
                if self.activation_pool.ptr is None:
                    raise TensorRTRuntimeError(
                        f"{self.engine_path.name}: shared TensorRT activation pool is closed"
                    )
                return self._run_impl(inputs)

    def _run_impl(self, inputs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        stream = self._stream()
        prepared_inputs: dict[str, np.ndarray] = {}

        if self.engine is None or self.context is None:
            raise TensorRTRuntimeError(f"{self.engine_path.name}: runner is closed")

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
            if self.engine.get_tensor_location(name) == self.trt.TensorLocation.DEVICE:
                ptr = self._buffer(name, arr.nbytes)
                memcpy_htod_async(ptr, arr, stream)
                self._set_tensor_address(name, ptr)
            else:
                self._set_tensor_address(name, arr.ctypes.data)

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
            outputs[name] = arr

            if self.engine.get_tensor_location(name) == self.trt.TensorLocation.DEVICE:
                ptr = self._buffer(name, arr.nbytes)
                self._set_tensor_address(name, ptr)
            else:
                self._set_tensor_address(name, arr.ctypes.data)

        if not self.context.execute_async_v3(stream_handle=stream.handle):
            raise TensorRTRuntimeError(
                f"{self.engine_path.name}: execute_async_v3 failed"
            )

        for name, arr in outputs.items():
            if self.engine.get_tensor_location(name) == self.trt.TensorLocation.DEVICE:
                ptr = self.buffers[name].ptr
                assert ptr is not None
                memcpy_dtoh_async(arr, ptr, stream)

        stream.synchronize()
        return outputs

    def close(self) -> None:
        run_lock = getattr(self, "_run_lock", None)
        if run_lock is None:
            self._close_impl()
            return

        with run_lock:
            self._close_impl()

    def _close_impl(self) -> None:
        buffers = getattr(self, "buffers", None)
        if buffers is not None:
            for buffer in buffers.values():
                buffer.free()
            buffers.clear()

        stream = getattr(self, "stream", None)
        if stream is not None:
            stream.close()
            self.stream = None

        self.context = None
        self.engine = None
        self.runtime = None
        self.activation_pool = None

    def __enter__(self) -> "TrtEngineRunner":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
