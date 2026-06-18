from __future__ import annotations

from pathlib import Path
import time

import torch

from .device import Runtime
from .models.checkpoint import CheckpointLoader, REPO_ID
from .models.pipeline import VoiceConversionPipeline
from .types import AudioInput, ConversionResult, ReferenceConditioning

SAMPLE_RATE = 24_000


class VoiceConverter:
    def __init__(
        self,
        *,
        pipeline: VoiceConversionPipeline,
        runtime: Runtime,
        target: ReferenceConditioning | None = None,
    ):
        self.pipeline = pipeline
        self.runtime = runtime
        self.target = target
        self._target_voice_cache_key: tuple[Path, int, int] | None = None
        self._target_voice_cache: ReferenceConditioning | None = None

    @classmethod
    def from_local(
        cls,
        checkpoint_dir: str | Path,
        device: str | torch.device = "auto",
    ) -> "VoiceConverter":
        runtime = Runtime.create(device)
        bundle = CheckpointLoader.from_local(checkpoint_dir)
        pipeline = CheckpointLoader.load_pipeline(bundle, runtime)
        target = CheckpointLoader.load_builtin_reference(bundle, runtime)
        return cls(pipeline=pipeline, runtime=runtime, target=target)

    @classmethod
    def from_pretrained(
        cls,
        device: str | torch.device = "auto",
        *,
        repo_id: str = REPO_ID,
    ) -> "VoiceConverter":
        runtime = Runtime.create(device)
        bundle = CheckpointLoader.from_pretrained(repo_id)
        pipeline = CheckpointLoader.load_pipeline(bundle, runtime)
        target = CheckpointLoader.load_builtin_reference(bundle, runtime)
        return cls(pipeline=pipeline, runtime=runtime, target=target)

    @torch.inference_mode()
    def set_target_voice(self, target_voice: AudioInput) -> None:
        if isinstance(target_voice, str | Path):
            path = Path(target_voice).expanduser().resolve(strict=False)
            stat = path.stat()
            cache_key = (path, stat.st_mtime_ns, stat.st_size)

            if cache_key == self._target_voice_cache_key:
                self.target = self._target_voice_cache
                return

            self.target = self.pipeline.encode_reference(path)
            self._target_voice_cache_key = cache_key
            self._target_voice_cache = self.target
            return

        self.target = self.pipeline.encode_reference(target_voice)
        self._target_voice_cache_key = None
        self._target_voice_cache = None

    @torch.inference_mode()
    def convert(
        self,
        source: AudioInput,
        *,
        target_voice: AudioInput | None = None,
        steps: int | None = None,
        return_cpu: bool = True,
        generator: torch.Generator | None = None,
    ) -> ConversionResult:
        start = time.perf_counter()

        if target_voice is not None:
            ref = self.pipeline.encode_reference(target_voice)
        else:
            if self.target is None:
                raise ValueError("Set a target voice before conversion.")
            ref = self.target

        wav = self.pipeline.convert(source, ref, steps=steps, generator=generator)
        if return_cpu:
            wav = wav.detach().cpu()

        total = time.perf_counter() - start
        duration = wav.shape[-1] / SAMPLE_RATE
        return ConversionResult(
            waveform=wav,
            sample_rate=SAMPLE_RATE,
            timings={
                "wall_total": total,
                "total": total,
                "audio_duration_sec": duration,
                "rtf": total / duration if duration > 0 else 0.0,
            },
        )
