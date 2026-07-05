from __future__ import annotations

import threading

from .cuda import DeviceBuffer
from .errors import TensorRTRuntimeError


class TrtActivationMemoryPool:
    def __init__(self, nbytes: int):
        requested_nbytes = int(nbytes)
        if requested_nbytes < 0:
            raise TensorRTRuntimeError(
                f"TensorRT activation memory size must be non-negative; got {requested_nbytes}"
            )

        self.nbytes = max(1, requested_nbytes)
        self.buffer = DeviceBuffer()
        self.ptr: int | None = self.buffer.ensure_capacity(self.nbytes)
        self.lock = threading.RLock()

    def close(self) -> None:
        self.buffer.free()
        self.ptr = None

    def __enter__(self) -> "TrtActivationMemoryPool":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
