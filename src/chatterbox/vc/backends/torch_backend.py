from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from ...audio import DEC_COND_LEN, S3GEN_SR
from ...models.s3gen import S3Gen
from ...models.s3gen.checkpoint_conversion import \
    convert_diffusers_transformer_keys
from ..conditioning import VoiceConditionTensors
from ..errors import VoiceConditioningError
from ..preprocess import load_wav_16k, load_wav_24k
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


def download_pretrained_checkpoint(repo_id: str = REPO_ID) -> Path:
    from huggingface_hub import hf_hub_download

    local_path = None
    for fpath in ["s3gen_meanflow.safetensors", "conds.pt"]:
        local_path = hf_hub_download(repo_id=repo_id, filename=fpath)
    if local_path is None:
        raise FileNotFoundError("No checkpoint files were downloaded.")
    return Path(local_path).parent


class TorchVCBackend:
    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict | VoiceConditionTensors | None = None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self._target_voice_cache: VoiceConditionTensors | None = None
        self.ref_dict = None

        if ref_dict is not None:
            if isinstance(ref_dict, VoiceConditionTensors):
                self._target_voice_cache = ref_dict
                self.ref_dict = ref_dict.to_torch(
                    device=self.device, dtype=self.s3gen.dtype
                )
            else:
                self.ref_dict = self.s3gen.prepare_ref_dict(ref_dict)
                np_condition = {
                    k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
                    for k, v in self.ref_dict.items()
                    if v is not None
                }
                self._target_voice_cache = VoiceConditionTensors.from_mapping(
                    np_condition
                )

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        compile: bool = False,
    ) -> "TorchVCBackend":
        ckpt_dir = Path(ckpt_dir)
        map_location = torch.device("cpu")

        ref_dict = None
        builtin_voice = ckpt_dir / "conds.pt"
        if builtin_voice.exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states["gen"]

        s3gen = S3Gen(meanflow=True)
        state = load_file(ckpt_dir / "s3gen_meanflow.safetensors")
        state = convert_diffusers_transformer_keys(state)
        s3gen.load_state_dict(state, strict=False)
        s3gen.to(device).eval()

        s3gen.mel2wav.optimize_for_inference()
        ref_dict = s3gen.prepare_ref_dict(ref_dict) if ref_dict is not None else None

        if _is_cuda_device(device):
            _configure_cuda_runtime(device)

        should_compile = compile or os.getenv("CHATTERBOX_COMPILE", "0").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

        if should_compile:
            s3gen.compile_for_inference()

        if _is_cuda_device(device) or should_compile:
            s3gen.warmup(ref_dict=ref_dict)

        return cls(s3gen, device, ref_dict=ref_dict)

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

    def set_target_voice_from_tensors(
        self, target_voice: dict | VoiceConditionTensors
    ) -> None:
        condition = VoiceConditionTensors.from_mapping(target_voice)
        self._target_voice_cache = condition
        self.ref_dict = condition.to_torch(device=self.device, dtype=self.s3gen.dtype)

    def convert_from_path(
        self,
        audio_path: str | Path,
        target_voice_path: str | Path | None = None,
        profile: bool = False,
    ) -> VCResult:
        if target_voice_path:
            s3gen_ref_wav = load_wav_24k(
                target_voice_path, self.device, max_len=DEC_COND_LEN
            ).squeeze(0)
            self.ref_dict = self.s3gen.embed_ref(
                s3gen_ref_wav, S3GEN_SR, device=self.device
            )
            np_condition = {
                k: (v.detach().cpu().numpy() if torch.is_tensor(v) else v)
                for k, v in self.ref_dict.items()
                if v is not None
            }
            self._target_voice_cache = VoiceConditionTensors.from_mapping(np_condition)

        audio_16k = load_wav_16k(audio_path, self.device)
        return self.convert_from_tensors(audio_16k, self.ref_dict, profile)

    def convert_from_tensors(
        self,
        audio_16k: torch.Tensor,
        target_voice: dict | VoiceConditionTensors | None = None,
        profile: bool = False,
    ) -> VCResult:
        wall_start = time.perf_counter()
        active_sr = self.sr

        with torch.inference_mode():
            if target_voice is None:
                target_voice = self.ref_dict

            if target_voice is None:
                raise VoiceConditioningError("Target voice is not set.")

            if isinstance(target_voice, VoiceConditionTensors):
                ref_dict = target_voice.to_torch(
                    device=self.device, dtype=self.s3gen.dtype
                )
            else:
                ref_dict = self.s3gen.prepare_ref_dict(target_voice)

            audio_16k = audio_16k.to(self.device)
            if audio_16k.ndim == 1:
                audio_16k = audio_16k.unsqueeze(0)

            s3_tokens, _ = self.s3gen.tokenizer(audio_16k)
            output_mels = self.s3gen.flow_inference(
                speech_tokens=s3_tokens,
                ref_dict=ref_dict,
                finalize=True,
            )

            output_mels = output_mels.to(dtype=self.s3gen.dtype)
            output_wavs, _ = self.s3gen.hift_inference(output_mels, None)
            fade_len = min(output_wavs.size(1), self.s3gen.trim_fade.numel())
            output_wavs[:, :fade_len] *= self.s3gen.trim_fade[:fade_len]

            wav = output_wavs.detach().cpu()

        wall_total = time.perf_counter() - wall_start
        audio_duration = wav.shape[-1] / active_sr
        timings = {
            "wall_total": wall_total,
            "total": wall_total,
            "audio_duration_sec": audio_duration,
            "rtf": wall_total / audio_duration if audio_duration > 0 else 0,
        }
        return VCResult(wav=wav, sample_rate=active_sr, timings=timings)
