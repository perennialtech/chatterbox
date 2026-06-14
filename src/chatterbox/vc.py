from pathlib import Path
import time
from contextlib import contextmanager

import librosa
import torch
import perth
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.s3tokenizer import S3_SR
from .models.s3gen import S3GEN_SR, S3Gen

REPO_ID = "ResembleAI/chatterbox-turbo"


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
        self.watermarker = perth.PerthImplicitWatermarker()
        if ref_dict is None:
            self.ref_dict = None
        else:
            self.ref_dict = {
                k: v.to(device) if torch.is_tensor(v) else v
                for k, v in ref_dict.items()
            }

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
    def from_local(cls, ckpt_dir, device) -> "ChatterboxVC":
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device("cpu")
        else:
            map_location = None

        ref_dict = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            states = torch.load(builtin_voice, map_location=map_location)
            ref_dict = states["gen"]

        s3gen = S3Gen(meanflow=True)
        s3gen.load_state_dict(load_file(ckpt_dir / "s3gen_meanflow.safetensors"), strict=False)
        s3gen.to(device).eval()

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
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        s3gen_ref_wav = s3gen_ref_wav[: self.DEC_COND_LEN]
        self.ref_dict = self.s3gen.embed_ref(
            s3gen_ref_wav, S3GEN_SR, device=self.device
        )

    def generate(
        self,
        audio,
        target_voice_path=None,
    ):
        timings = {}

        with self._track_time(timings, "total"):
            if target_voice_path:
                with self._track_time(timings, "target_voice_setup"):
                    self.set_target_voice(target_voice_path)
            else:
                assert (
                    self.ref_dict is not None
                ), "Please `prepare_conditionals` first or specify `target_voice_path`"

            with torch.inference_mode():
                with self._track_time(timings, "audio_load"):
                    audio_16, _ = librosa.load(audio, sr=S3_SR)
                    audio_16 = torch.from_numpy(audio_16).float().to(self.device)[None,]

                with self._track_time(timings, "tokenize"):
                    s3_tokens, _ = self.s3gen.tokenizer(audio_16)

                with self._track_time(timings, "flow_inference"):
                    output_mels = self.s3gen.flow_inference(
                        speech_tokens=s3_tokens, ref_dict=self.ref_dict, finalize=True
                    )

                with self._track_time(timings, "vocoder_inference"):
                    output_mels = output_mels.to(dtype=self.s3gen.dtype)
                    output_wavs, _ = self.s3gen.hift_inference(output_mels, None)
                    output_wavs[:, : len(self.s3gen.trim_fade)] *= self.s3gen.trim_fade
                    wav = output_wavs.squeeze(0).detach().cpu().numpy()

                with self._track_time(timings, "watermark"):
                    watermarked_wav = self.watermarker.apply_watermark(
                        wav, sample_rate=self.sr
                    )

        audio_duration = watermarked_wav.shape[-1] / self.sr
        timings["audio_duration_sec"] = audio_duration
        timings["rtf"] = timings["total"] / audio_duration if audio_duration > 0 else 0

        return torch.from_numpy(watermarked_wav).unsqueeze(0), timings
