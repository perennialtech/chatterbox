from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Runtime:
    device: torch.device
    compute_dtype: torch.dtype = torch.float32
    compile: bool = False
    allow_tf32: bool = True

    @classmethod
    def create(
        cls,
        device: str | torch.device = "auto",
        *,
        compute_dtype: torch.dtype = torch.float32,
        compile: bool | None = None,
        allow_tf32: bool = True,
    ) -> "Runtime":
        resolved = resolve_device(device)
        use_compile = compile if compile is not None else resolved.type == "cuda"

        runtime = cls(
            device=resolved,
            compute_dtype=compute_dtype,
            compile=use_compile,
            allow_tf32=allow_tf32,
        )
        runtime.configure()
        return runtime

    def configure(self) -> None:
        if self.device.type != "cuda":
            return

        torch.backends.cuda.matmul.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        if self.allow_tf32:
            torch.set_float32_matmul_precision("high")


def resolve_device(device: str | torch.device) -> torch.device:
    if isinstance(device, torch.device):
        return device

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if device == "mps" and not torch.backends.mps.is_available():
        return torch.device("cpu")

    return torch.device(device)
