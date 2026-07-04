from __future__ import annotations

import numpy as np
from cuda import cudart

from .errors import TensorRTRuntimeError


def check_cuda(result, message: str = "CUDA call failed"):
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise TensorRTRuntimeError(f"{message}: {err}")
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def cuda_runtime_version() -> str:
    version = check_cuda(cudart.cudaRuntimeGetVersion(), "cudaRuntimeGetVersion failed")
    major = version // 1000
    minor = (version % 1000) // 10
    return f"{major}.{minor}"


class CudaStream:
    def __init__(self):
        self.handle = check_cuda(cudart.cudaStreamCreate(), "cudaStreamCreate failed")

    def synchronize(self) -> None:
        check_cuda(
            cudart.cudaStreamSynchronize(self.handle), "cudaStreamSynchronize failed"
        )

    def close(self) -> None:
        if getattr(self, "handle", None) is not None:
            check_cuda(
                cudart.cudaStreamDestroy(self.handle), "cudaStreamDestroy failed"
            )
            self.handle = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class DeviceBuffer:
    def __init__(self):
        self.ptr: int | None = None
        self.capacity = 0

    def ensure_capacity(self, nbytes: int) -> int:
        if self.ptr is not None and self.capacity >= nbytes:
            return self.ptr
        self.free()
        self.ptr = int(check_cuda(cudart.cudaMalloc(nbytes), "cudaMalloc failed"))
        self.capacity = nbytes
        return self.ptr

    def free(self) -> None:
        if self.ptr is not None:
            check_cuda(cudart.cudaFree(self.ptr), "cudaFree failed")
            self.ptr = None
            self.capacity = 0

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass


def memcpy_htod_async(dst_ptr: int, src: np.ndarray, stream: CudaStream) -> None:
    check_cuda(
        cudart.cudaMemcpyAsync(
            dst_ptr,
            src.ctypes.data,
            int(src.nbytes),
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            stream.handle,
        ),
        "cudaMemcpyAsync HtoD failed",
    )


def memcpy_dtoh_async(dst: np.ndarray, src_ptr: int, stream: CudaStream) -> None:
    check_cuda(
        cudart.cudaMemcpyAsync(
            dst.ctypes.data,
            src_ptr,
            int(dst.nbytes),
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
            stream.handle,
        ),
        "cudaMemcpyAsync DtoH failed",
    )
