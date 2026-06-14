from pathlib import Path
import time
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
import threading

import numpy as np
import librosa
import torch
import perth
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR, S3_TOKEN_RATE
from .models.s3gen import S3GEN_SR, S3Gen

REPO_ID = "ResembleAI/chatterbox"


class ChatterboxVC:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR
    TARGET_VOICE_CACHE_SIZE = 16

    def __init__(
        self,
        s3gen: S3Gen,
        device: str,
        ref_dict: dict = None,
        precision: str = "fp32",
        target_voice_cache_size: int = TARGET_VOICE_CACHE_SIZE,
    ):
        self.sr = S3GEN_SR
        self.s3gen = s3gen
        self.device = device
        self.watermarker = perth.PerthImplicitWatermarker()
        self.precision = self._normalize_precision(device, precision)
        self._autocast_dtype = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }.get(self.precision)
        self._target_voice_cache_size = target_voice_cache_size
        self._target_voice_cache = OrderedDict()
        self._target_voice_cache_lock = threading.Lock()
        self.compiled = False
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

    @staticmethod
    def configure_backend(device):
        device_type = str(device).lower()
        if "cuda" not in device_type or not torch.cuda.is_available():
            return

        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    @staticmethod
    def _normalize_precision(device, precision):
        precision = precision.lower()
        if precision in ["float32", "fp32"]:
            return "fp32"
        if precision in ["float16", "fp16"]:
            return "fp16" if "cuda" in str(device).lower() else "fp32"
        if precision in ["bfloat16", "bf16"]:
            if "cuda" in str(device).lower() and torch.cuda.is_bf16_supported():
                return "bf16"
            return "fp32"
        raise ValueError(f"Unsupported precision: {precision}")

    def _autocast(self):
        if self._autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self._autocast_dtype)

    def _sync(self):
        device_type = str(self.device).lower()
        if "cuda" in device_type and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif "mps" in device_type and torch.backends.mps.is_available():
            torch.mps.synchronize()

    @contextmanager
    def _track_time(self, timings_dict, key):
        self._sync()
        start = time.perf_counter()
        yield
        self._sync()
        timings_dict[key] = time.perf_counter() - start

    @classmethod
    def from_local(
        cls,
        ckpt_dir,
        device,
        precision: str = "fp32",
        compile_model: bool = False,
        warmup: bool = False,
    ) -> "ChatterboxVC":
        ckpt_dir = Path(ckpt_dir)
        cls.configure_backend(device)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states["gen"]

        s3gen = S3Gen()
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen.safetensors"), strict=False)
        s3gen.to(device).eval()

        instance = cls(s3gen, device, ref_dict=ref_dict, precision=precision)
        if compile_model:
            instance.compile()
        if warmup:
            instance.warmup()
        return instance

    @classmethod
    def from_pretrained(
        cls,
        device,
        precision: str = "fp32",
        compile_model: bool = False,
        warmup: bool = False,
    ) -> "ChatterboxVC":
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

        for fpath in ["s3gen.safetensors", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(
            Path(local_path).parent,
            device,
            precision=precision,
            compile_model=compile_model,
            warmup=warmup,
        )

    def compile(self):
        if self.compiled:
            return
        if not hasattr(torch, "compile"):
            return

        self.s3gen.flow.decoder.estimator = torch.compile(
            self.s3gen.flow.decoder.estimator,
            dynamic=True,
        )
        self.s3gen.mel2wav = torch.compile(
            self.s3gen.mel2wav,
            dynamic=True,
        )
        self.compiled = True

    def _target_voice_cache_key(self, wav_fpath):
        path = Path(wav_fpath).expanduser().resolve()
        stat = path.stat()
        return str(path), stat.st_mtime_ns, stat.st_size

    def _embed_target_voice(self, wav_fpath):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        return self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

    def get_target_voice(self, wav_fpath):
        key = self._target_voice_cache_key(wav_fpath)

        with self._target_voice_cache_lock:
            cached = self._target_voice_cache.get(key)
            if cached is not None:
                self._target_voice_cache.move_to_end(key)
                return cached

        ref_dict = self._embed_target_voice(wav_fpath)

        with self._target_voice_cache_lock:
            cached = self._target_voice_cache.get(key)
            if cached is not None:
                self._target_voice_cache.move_to_end(key)
                return cached

            self._target_voice_cache[key] = ref_dict
            while len(self._target_voice_cache) > self._target_voice_cache_size:
                self._target_voice_cache.popitem(last=False)

        return ref_dict

    def set_target_voice(self, wav_fpath):
        self.ref_dict = self.get_target_voice(wav_fpath)
        return self.ref_dict

    @torch.inference_mode()
    def warmup(
        self, seconds: float = 1.0, n_cfm_timesteps: int = 10, cfg_rate: float = 0.7
    ):
        if self.ref_dict is None:
            return

        dummy_audio = np.zeros(int(max(0.1, seconds) * S3_SR), dtype=np.float32)

        self.generate(
            audio=dummy_audio,
            n_cfm_timesteps=n_cfm_timesteps,
            cfg_rate=cfg_rate,
        )

    def generate(
        self,
        audio,
        target_voice_path=None,
        n_cfm_timesteps=None,
        cfg_rate=None,
    ):
        timings = {}

        with self._track_time(timings, "total"):
            if target_voice_path:
                with self._track_time(timings, "target_voice_setup"):
                    ref_dict = self.get_target_voice(target_voice_path)
            else:
                assert (
                    self.ref_dict is not None
                ), "Please `prepare_conditionals` first or specify `target_voice_path`"
                ref_dict = self.ref_dict

            with torch.inference_mode():
                with self._track_time(timings, "audio_load"):
                    if isinstance(audio, np.ndarray):
                        audio_16 = audio
                    else:
                        audio_16, _ = librosa.load(audio, sr=S3_SR)
                    audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None,]

                with self._track_time(timings, "tokenize"):
                    s3_tokens, _ = self.s3gen.tokenizer(audio_16)

                with self._track_time(timings, "flow_inference"):
                    with self._autocast():
                        output_mels = self.s3gen.flow_inference(
                            speech_tokens=s3_tokens,
                            ref_dict=ref_dict,
                            finalize=True,
                            n_cfm_timesteps=n_cfm_timesteps,
                            cfg_rate=cfg_rate,
                        )

                with self._track_time(timings, "vocoder_inference"):
                    with self._autocast():
                        output_mels = output_mels.to(dtype=self.s3gen.dtype)
                        output_wavs, _ = self.s3gen.hift_inference(output_mels, None)
                        output_wavs[
                            :, : len(self.s3gen.trim_fade)
                        ] *= self.s3gen.trim_fade
                    wav = output_wavs.squeeze(0).detach().float().cpu().numpy()

                with self._track_time(timings, "watermark"):
                    watermarked_wav = self.watermarker.apply_watermark(
                        wav, sample_rate=self.sr
                    )

        audio_duration = watermarked_wav.shape[-1] / self.sr
        timings["audio_duration_sec"] = audio_duration
        timings["rtf"] = timings["total"] / audio_duration if audio_duration > 0 else 0

        return torch.from_numpy(watermarked_wav).unsqueeze(0), timings
