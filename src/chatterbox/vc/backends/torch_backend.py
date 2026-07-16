from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from ...audio import S3_SR, S3GEN_SR, load_audio_mono
from ...models.s3gen import S3Gen
from ...models.s3gen.checkpoint_conversion import \
    convert_diffusers_transformer_keys
from ...models.s3gen.conditioning import (ConditioningError,
                                          S3ReferenceCondition)
from ..errors import VoiceConditioningError
from ..types import VCResult

logger = logging.getLogger(__name__)

REPO_ID = "ResembleAI/chatterbox-turbo"


def _is_cuda_device(device) -> bool:
    return torch.device(device).type == "cuda"


def _configure_cuda_runtime(device) -> None:
    if not _is_cuda_device(device):
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def load_wav_24k(
    path: str | Path,
    device,
    max_len: int | None = None,
) -> torch.Tensor:
    return load_audio_mono(path, S3GEN_SR, device, max_len=max_len)


def load_wav_16k(path: str | Path, device) -> torch.Tensor:
    return load_audio_mono(path, S3_SR, device)


def download_pretrained_checkpoint(repo_id: str = REPO_ID) -> Path:
    from huggingface_hub import snapshot_download

    checkpoint_dir = Path(
        snapshot_download(
            repo_id=repo_id,
            allow_patterns=["s3gen_meanflow.safetensors", "conds.pt"],
        )
    )
    checkpoint_path = checkpoint_dir / "s3gen_meanflow.safetensors"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Required checkpoint file was not downloaded: {checkpoint_path.name}"
        )
    return checkpoint_dir


class TorchVCBackend:
    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_condition: dict | S3ReferenceCondition | None = None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.ref_condition: S3ReferenceCondition | None = None

        if ref_condition is not None:
            try:
                self.ref_condition = self._prepare_ref_condition(ref_condition)
            except ConditioningError as exc:
                raise VoiceConditioningError(str(exc)) from exc

    def _prepare_ref_condition(
        self,
        target_voice: dict | S3ReferenceCondition,
    ) -> S3ReferenceCondition:
        return self.s3gen.prepare_ref_condition(target_voice)

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        compile: bool = False,
    ) -> "TorchVCBackend":
        ckpt_dir = Path(ckpt_dir)
        map_location = torch.device("cpu")

        builtin_ref_condition = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            states = torch.load(builtin_voice, map_location=map_location)
            builtin_ref_condition = states["gen"]

        s3gen = S3Gen()
        state = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        state = convert_diffusers_transformer_keys(state)
        s3gen.load_state_dict(state, strict=True)
        s3gen.to(device).eval()

        s3gen.mel2wav.optimize_for_inference()

        if _is_cuda_device(device):
            _configure_cuda_runtime(device)

        backend = cls(s3gen, device, ref_condition=builtin_ref_condition)

        should_compile = compile or os.getenv("CHATTERBOX_COMPILE", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        if should_compile:
            s3gen.compile_for_inference()

        if _is_cuda_device(device) or should_compile:
            s3gen.warmup(ref_dict=backend.ref_condition)

        return backend

    @classmethod
    def from_pretrained(
        cls,
        device,
        compile: bool = False,
    ) -> "TorchVCBackend":
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                logger.warning(
                    "MPS unavailable because this PyTorch install was not built with MPS."
                )
            else:
                logger.warning("MPS unavailable on this macOS/device combination.")
            device = "cpu"

        return cls.from_local(
            download_pretrained_checkpoint(),
            device,
            compile=compile,
        )

    def set_target_voice_condition(
        self,
        target_voice: dict | S3ReferenceCondition,
    ) -> None:
        try:
            ref_condition = self._prepare_ref_condition(target_voice)
        except ConditioningError as exc:
            raise VoiceConditioningError(str(exc)) from exc

        self.ref_condition = ref_condition

    def convert_from_path(
        self,
        audio_path: str | Path,
        target_voice_path: str | Path | None = None,
        profile: bool = False,
    ) -> VCResult:
        if target_voice_path:
            try:
                s3gen_ref_wav = load_wav_24k(target_voice_path, self.device)
                ref_condition = self.s3gen.embed_ref(
                    s3gen_ref_wav,
                    S3GEN_SR,
                    device=self.device,
                )
            except ConditioningError as exc:
                raise VoiceConditioningError(str(exc)) from exc

            self.ref_condition = ref_condition

        audio_16k = load_wav_16k(audio_path, self.device)
        if audio_16k.ndim == 1:
            audio_16k = audio_16k.unsqueeze(0)
        return self.convert_from_tensors(audio_16k, self.ref_condition, profile)

    def convert_from_tensors(
        self,
        audio_16k: torch.Tensor,
        target_voice: dict | S3ReferenceCondition | None = None,
        profile: bool = False,
    ) -> VCResult:
        wall_start = time.perf_counter()
        active_sr = self.sr

        try:
            with torch.inference_mode():
                if target_voice is None:
                    ref_condition = self.ref_condition
                else:
                    ref_condition = self._prepare_ref_condition(target_voice)

                if ref_condition is None:
                    raise VoiceConditioningError("Target voice is not set.")

                audio_16k = audio_16k.to(self.device)
                if audio_16k.ndim == 1:
                    audio_16k = audio_16k.unsqueeze(0)

                s3_tokens, s3_token_lens = self.s3gen.tokenizer(audio_16k)
                output_wavs, _ = self.s3gen.inference(
                    speech_tokens=s3_tokens,
                    speech_token_lens=s3_token_lens,
                    ref_dict=ref_condition,
                    drop_invalid_tokens=False,
                )

                wav = output_wavs.detach().cpu()
        except ConditioningError as exc:
            raise VoiceConditioningError(str(exc)) from exc

        wall_total = time.perf_counter() - wall_start
        audio_duration = wav.shape[-1] / active_sr
        timings = {
            "wall_total": wall_total,
            "total": wall_total,
            "audio_duration_sec": audio_duration,
            "rtf": wall_total / audio_duration if audio_duration > 0 else 0,
        }
        return VCResult(wav=wav, sample_rate=active_sr, timings=timings)
