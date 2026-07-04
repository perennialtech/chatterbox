import logging
import os
import time
from pathlib import Path

import torch
from safetensors.torch import load_file

from ..audio import DEC_COND_LEN, S3_SR, S3GEN_SR, load_audio_mono
from ..models.s3gen import S3Gen
from ..models.s3gen.checkpoint_conversion import \
    convert_diffusers_transformer_keys
from .errors import BackendUnavailableError, VoiceConditioningError
from .types import VCBackend, VCResult

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
        ref_dict: dict | None = None,
        flowhigh=None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self._target_voice_cache = None
        self.ref_dict = (
            None if ref_dict is None else self.s3gen.prepare_ref_dict(ref_dict)
        )
        self.flowhigh = flowhigh

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        load_flowhigh: bool = True,
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

        flowhigh = None
        if load_flowhigh:
            from flowhigh.flowhighsr import FlowHighSR

            flowhigh = FlowHighSR.from_pretrained(device=device)

        return cls(s3gen, device, ref_dict=ref_dict, flowhigh=flowhigh)

    @classmethod
    def from_pretrained(
        cls,
        device,
        load_flowhigh: bool = True,
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
            load_flowhigh=load_flowhigh,
            compile=compile,
        )

    def set_target_voice_from_tensors(self, target_voice: dict) -> None:
        self._target_voice_cache = target_voice
        self.ref_dict = target_voice

    def convert_from_path(
        self,
        audio_path: str | Path,
        target_voice_path: str | Path | None = None,
        profile: bool = False,
        upscale: bool = False,
    ) -> VCResult:
        if target_voice_path:
            s3gen_ref_wav = load_audio_mono(
                target_voice_path, S3GEN_SR, self.device, max_len=DEC_COND_LEN
            )
            self.ref_dict = self.s3gen.embed_ref(
                s3gen_ref_wav, S3GEN_SR, device=self.device
            )
            self._target_voice_cache = self.ref_dict

        audio_16k = load_audio_mono(audio_path, S3_SR, self.device).unsqueeze(0)
        return self.convert_from_tensors(audio_16k, self.ref_dict, profile, upscale)

    def convert_from_tensors(
        self,
        audio_16k: torch.Tensor,
        target_voice: dict | None = None,
        profile: bool = False,
        upscale: bool = False,
    ) -> VCResult:
        if upscale and self.flowhigh is None:
            raise BackendUnavailableError(
                "FlowHigh model is not loaded. Initialize with load_flowhigh=True."
            )

        wall_start = time.perf_counter()
        active_sr = self.sr

        with torch.inference_mode():
            if target_voice is None:
                raise VoiceConditioningError("Target voice is not set.")

            s3_tokens, _ = self.s3gen.tokenizer(audio_16k)
            output_mels = self.s3gen.flow_inference(
                speech_tokens=s3_tokens,
                ref_dict=target_voice,
                finalize=True,
            )

            output_mels = output_mels.to(dtype=self.s3gen.dtype)
            output_wavs, _ = self.s3gen.hift_inference(output_mels, None)
            output_wavs[:, : len(self.s3gen.trim_fade)] *= self.s3gen.trim_fade

            if upscale:
                output_wavs = self.flowhigh.enhance(output_wavs, sample_rate=active_sr)
                output_wavs = (
                    output_wavs.unsqueeze(0) if output_wavs.ndim == 1 else output_wavs
                )
                active_sr = self.flowhigh.codec.sampling_rate

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


class ChatterboxVC:
    def __init__(self, backend: VCBackend):
        self.backend = backend
        self.sr = backend.sr

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        load_flowhigh: bool = True,
        compile: bool = False,
    ) -> "ChatterboxVC":
        return cls(
            TorchVCBackend.from_local(
                ckpt_dir,
                device,
                load_flowhigh=load_flowhigh,
                compile=compile,
            )
        )

    @classmethod
    def from_pretrained(
        cls,
        device,
        load_flowhigh: bool = True,
        compile: bool = False,
    ) -> "ChatterboxVC":
        return cls(
            TorchVCBackend.from_pretrained(
                device,
                load_flowhigh=load_flowhigh,
                compile=compile,
            )
        )

    @classmethod
    def from_onnx_artifacts(
        cls, artifact_dir: str | Path, providers: list[str] | None = None
    ):
        from ..onnx_export.runtime.sessions import OnnxSessions
        from ..onnx_export.runtime.vc import OnnxVCBackend

        return cls(
            OnnxVCBackend(
                OnnxSessions.from_dir(Path(artifact_dir), providers=providers)
            )
        )

    def generate(
        self,
        audio,
        target_voice_path=None,
        profile: bool = False,
        upscale: bool = False,
    ):
        if isinstance(audio, (str, Path)):
            result = self.backend.convert_from_path(
                audio,
                target_voice_path=target_voice_path,
                profile=profile,
                upscale=upscale,
            )
        else:
            result = self.backend.convert_from_tensors(
                audio,
                target_voice=None,  # if user passes tensors, they need to have used set_target_voice_from_tensors previously (or modified kwargs directly)
                profile=profile,
                upscale=upscale,
            )

        return result.wav, result.sample_rate, result.timings
