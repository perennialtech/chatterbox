from contextlib import nullcontext
from pathlib import Path
import time

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .audio import load_audio_mono
from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen
from .timing import InferenceTimer

REPO_ID = "ResembleAI/chatterbox-turbo"


def _is_cuda_device(device) -> bool:
    return torch.device(device).type == "cuda"


def _configure_cuda_runtime(device) -> None:
    if not _is_cuda_device(device):
        return

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def _track(timer, key):
    return timer.track(key) if timer is not None else nullcontext()


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict = None,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self._target_voice_cache_key = None
        self._target_voice_cache = None
        self.ref_dict = (
            None if ref_dict is None else self.s3gen.prepare_ref_dict(ref_dict)
        )

    @classmethod
    def from_local(cls, ckpt_dir, device) -> "ChatterboxVC":
        ckpt_dir = Path(ckpt_dir)
        map_location = torch.device("cpu")

        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states["gen"]

        s3gen = S3Gen(meanflow=True)
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen_meanflow.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        s3gen.mel2wav.optimize_for_inference()
        ref_dict = s3gen.prepare_ref_dict(ref_dict) if ref_dict is not None else None

        if _is_cuda_device(device):
            _configure_cuda_runtime(device)
            s3gen.compile_for_inference()
            s3gen.warmup(ref_dict=ref_dict)

        return cls(s3gen, device, ref_dict=ref_dict)

    @classmethod
    def from_pretrained(cls, device) -> "ChatterboxVC":
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
                )
            device = "cpu"

        for fpath in ["s3gen_meanflow.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def set_target_voice(self, wav_fpath):
        wav_fpath = Path(wav_fpath).expanduser().resolve(strict=False)
        wav_stat = wav_fpath.stat()
        cache_key = (wav_fpath, wav_stat.st_mtime_ns, wav_stat.st_size)

        if self._target_voice_cache_key == cache_key:
            self.ref_dict = self._target_voice_cache
            return

        s3gen_ref_wav = load_audio_mono(
            wav_fpath,
            S3GEN_SR,
            self.device,
            max_len=self.DEC_COND_LEN,
        )
        self.ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device
        )
        self._target_voice_cache_key = cache_key
        self._target_voice_cache = self.ref_dict

    def generate(
        self,
        audio,
        target_voice_path=None,
        profile: bool = False,
    ):
        timings = {}
        timer = InferenceTimer(timings, self.device) if profile else None
        wall_start = time.perf_counter()

        with _track(timer, "total"):
            if target_voice_path:
                with _track(timer, "target_voice_setup"):
                    self.set_target_voice(target_voice_path)
            else:
                assert (
                    self.ref_dict is not None
                ), "Please `prepare_conditionals` first or specify `target_voice_path`"

            with torch.inference_mode():
                with _track(timer, "audio_load"):
                    audio_16 = load_audio_mono(audio, S3_SR, self.device).unsqueeze(0)

                with _track(timer, "tokenize"):
                    s3_tokens, _ = self.s3gen.tokenizer(audio_16)

                with _track(timer, "flow_inference"):
                    output_mels = self.s3gen.flow_inference(
                        speech_tokens=s3_tokens,
                        ref_dict=self.ref_dict,
                        finalize=True,
                        timer=timer.child("flow") if timer is not None else None,
                    )

                with _track(timer, "vocoder_inference"):
                    with _track(timer, "vocoder.prepare_mels"):
                        output_mels = output_mels.to(dtype=self.s3gen.dtype)

                    with _track(timer, "vocoder.hift_inference"):
                        output_wavs, _ = self.s3gen.hift_inference(
                            output_mels,
                            None,
                            timer=timer.child("vocoder") if timer is not None else None,
                        )

                    with _track(timer, "vocoder.trim_fade"):
                        output_wavs[
                            :, : len(self.s3gen.trim_fade)
                        ] *= self.s3gen.trim_fade

                    with _track(timer, "vocoder.to_cpu"):
                        wav = output_wavs.squeeze(0).detach().cpu().numpy()

        if timer is not None:
            timer.finalize()

        wall_total = time.perf_counter() - wall_start
        audio_duration = wav.shape[-1] / self.sr
        timings["wall_total"] = wall_total
        timings["total"] = wall_total
        timings["audio_duration_sec"] = audio_duration
        timings["rtf"] = wall_total / audio_duration if audio_duration > 0 else 0

        return torch.from_numpy(wav).unsqueeze(0), timings
