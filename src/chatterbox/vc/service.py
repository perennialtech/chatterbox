from __future__ import annotations

from pathlib import Path

from .backends.torch_backend import TorchVCBackend
from .conditioning import VoiceConditionTensors
from .types import VCBackend


class ChatterboxVC:
    def __init__(self, backend: VCBackend):
        self.backend = backend
        self.sr = backend.sr

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        compile: bool = False,
    ) -> "ChatterboxVC":
        return cls(
            TorchVCBackend.from_local(
                ckpt_dir,
                device,
                compile=compile,
            )
        )

    @classmethod
    def from_pretrained(
        cls,
        device,
        compile: bool = False,
    ) -> "ChatterboxVC":
        return cls(
            TorchVCBackend.from_pretrained(
                device,
                compile=compile,
            )
        )

    def set_target_voice_from_tensors(
        self, target_voice: dict | VoiceConditionTensors
    ) -> None:
        self.backend.set_target_voice_from_tensors(target_voice)

    def generate(
        self,
        audio,
        target_voice_path=None,
        profile: bool = False,
    ):
        if isinstance(audio, (str, Path)):
            result = self.backend.convert_from_path(
                audio,
                target_voice_path=target_voice_path,
                profile=profile,
            )
        else:
            result = self.backend.convert_from_tensors(
                audio,
                target_voice=None,
                profile=profile,
            )

        return result.wav, result.sample_rate, result.timings
