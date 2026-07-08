from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch

from ...audio import DEC_COND_LEN, S3_SR, S3GEN_SR, resample_audio
from ...onnx_export.constants import (GRAPH_FLOW_DECODER_MEANFLOW2,
                                      GRAPH_REFERENCE_MEL_24K,
                                      GRAPH_S3_TOKENIZER_QUANTIZER,
                                      GRAPH_SPEAKER_ENCODER, GRAPH_TOKEN_TO_MU,
                                      GRAPH_VOCODER_HIFT)
from ...tensorrt.engine import TrtEngineRunner
from ...tensorrt.manifest import load_trt_manifest
from ...tensorrt.memory import TrtActivationMemoryPool
from ..conditioning import VoiceConditionTensors
from ..errors import BackendUnavailableError, VoiceConditioningError
from ..postprocess import apply_initial_trim_fade
from ..preprocess import (compute_fbank, compute_s3_log_mel, load_wav_16k,
                          load_wav_24k)
from ..types import VCResult

logger = logging.getLogger(__name__)

REQUIRED_TRT_VC_GRAPHS: tuple[str, ...] = (
    GRAPH_REFERENCE_MEL_24K,
    GRAPH_SPEAKER_ENCODER,
    GRAPH_S3_TOKENIZER_QUANTIZER,
    GRAPH_TOKEN_TO_MU,
    GRAPH_FLOW_DECODER_MEANFLOW2,
    GRAPH_VOCODER_HIFT,
)


def _format_mib(nbytes: int) -> str:
    return f"{nbytes / 1024**2:.2f} MiB"


def _log_activation_memory_plan(sizes: dict[str, int]) -> None:
    total = sum(sizes.values())
    shared = max(sizes.values()) if sizes else 0
    saved_pct = (1.0 - (shared / total)) * 100.0 if total > 0 else 0.0

    per_engine = ", ".join(
        f"{graph_name}={_format_mib(nbytes)}" for graph_name, nbytes in sizes.items()
    )
    logger.info("TensorRT VC activation memory by engine: %s", per_engine)
    logger.info(
        "TensorRT VC shared activation memory plan: sum=%s shared_pool=%s theoretical_saved=%.1f%%",
        _format_mib(total),
        _format_mib(shared),
        saved_pct,
    )


class TensorRTVCBackend:
    sr: int = S3GEN_SR

    def __init__(
        self,
        engine_dir: Path,
        manifest: dict,
        runners: dict[str, TrtEngineRunner],
        activation_pool: TrtActivationMemoryPool,
    ):
        self.engine_dir = Path(engine_dir)
        self.manifest = manifest
        self.runners = runners
        self.activation_pool: TrtActivationMemoryPool | None = activation_pool
        self.constants = manifest["constants"]
        self.source_hop = int(self.constants["source_hop"])
        self.trim_fade_len = int(self.constants["trim_fade_len"])
        self._target_voice_cache: VoiceConditionTensors | None = None

    @classmethod
    def from_engine_dir(cls, engine_dir: str | Path) -> "TensorRTVCBackend":
        engine_dir = Path(engine_dir)
        manifest = load_trt_manifest(engine_dir)
        engine_entries = manifest["engines"]

        missing = [
            graph_name
            for graph_name in REQUIRED_TRT_VC_GRAPHS
            if graph_name not in engine_entries
        ]
        if missing:
            raise BackendUnavailableError(f"Missing TensorRT engines: {missing}")

        activation_sizes = {
            graph_name: TrtEngineRunner.inspect_activation_memory(
                engine_dir / engine_entries[graph_name]["engine"]
            )
            for graph_name in REQUIRED_TRT_VC_GRAPHS
        }
        _log_activation_memory_plan(activation_sizes)

        activation_pool = TrtActivationMemoryPool(max(activation_sizes.values()))
        runners: dict[str, TrtEngineRunner] = {}

        try:
            for graph_name in REQUIRED_TRT_VC_GRAPHS:
                entry = engine_entries[graph_name]
                runners[graph_name] = TrtEngineRunner(
                    engine_dir / entry["engine"],
                    activation_pool=activation_pool,
                )
        except Exception:
            for runner in runners.values():
                runner.close()
            activation_pool.close()
            raise

        return cls(engine_dir, manifest, runners, activation_pool)

    def _runner(self, graph_name: str) -> TrtEngineRunner:
        if graph_name not in self.runners:
            raise BackendUnavailableError(
                f"Required TensorRT engine {graph_name} is missing"
            )
        return self.runners[graph_name]

    def set_target_voice_from_tensors(
        self, target_voice: dict | VoiceConditionTensors
    ) -> None:
        self._target_voice_cache = VoiceConditionTensors.from_mapping(target_voice)

    def convert_from_path(
        self,
        audio_path: str | Path,
        target_voice_path: str | Path | None = None,
        profile: bool = False,
    ) -> VCResult:
        if target_voice_path:
            ref_wav_24k = load_wav_24k(target_voice_path, "cpu", max_len=DEC_COND_LEN)
            ref_wav_16k = resample_audio(ref_wav_24k, S3GEN_SR, S3_SR, "cpu")
            self.set_target_voice_from_tensors(
                self._extract_target_voice_tensors(
                    ref_wav_24k.numpy(), ref_wav_16k.numpy()
                )
            )

        audio_16k = load_wav_16k(audio_path, "cpu")
        return self.convert_from_tensors(audio_16k, self._target_voice_cache, profile)

    def _tokenize_audio(self, audio_16k: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        log_mel, mel_lengths = compute_s3_log_mel(audio_16k)
        out = self._runner(GRAPH_S3_TOKENIZER_QUANTIZER).run(
            {"log_mel": log_mel, "mel_lengths": mel_lengths.astype(np.int32)}
        )
        return out["speech_tokens"].astype(np.int64), out[
            "speech_token_lengths"
        ].astype(np.int64)

    def _extract_target_voice_tensors(
        self, ref_wav_24k: np.ndarray, ref_wav_16k: np.ndarray
    ) -> VoiceConditionTensors:
        mel_out = self._runner(GRAPH_REFERENCE_MEL_24K).run(
            {"wav_24k": np.ascontiguousarray(ref_wav_24k.astype(np.float32))}
        )
        prompt_feat = mel_out["prompt_feat"].astype(np.float32)

        fbank = compute_fbank(torch.from_numpy(ref_wav_16k))
        speaker_out = self._runner(GRAPH_SPEAKER_ENCODER).run({"fbank": fbank})
        embedding = speaker_out["embedding"].astype(np.float32)

        prompt_token, prompt_token_len = self._tokenize_audio(
            torch.from_numpy(ref_wav_16k)
        )
        if prompt_feat.shape[1] != 2 * prompt_token.shape[1]:
            target_len = min(prompt_token.shape[1], prompt_feat.shape[1] // 2)
            if target_len <= 0:
                raise VoiceConditioningError(
                    "reference audio is too short for aligned conditioning"
                )
            prompt_token = prompt_token[:, :target_len]
            prompt_token_len = np.array([target_len], dtype=np.int64)
            prompt_feat = prompt_feat[:, : 2 * target_len, :]

        return VoiceConditionTensors.from_mapping(
            {
                "prompt_token": prompt_token,
                "prompt_token_len": prompt_token_len,
                "prompt_feat": prompt_feat,
                "embedding": embedding,
            }
        )

    def convert_from_tensors(
        self,
        audio_16k: torch.Tensor,
        target_voice: dict | VoiceConditionTensors | None = None,
        profile: bool = False,
    ) -> VCResult:
        wall_start = time.perf_counter()

        if target_voice is None:
            target_voice = self._target_voice_cache
        if target_voice is None:
            raise VoiceConditioningError("Target voice is not set.")
        condition = VoiceConditionTensors.from_mapping(target_voice)

        speech_tokens, speech_token_lens = self._tokenize_audio(audio_16k)
        wav, _ = self._convert_from_tokens(speech_tokens, speech_token_lens, condition)

        wall_total = time.perf_counter() - wall_start
        audio_duration = wav.shape[-1] / self.sr
        timings = {
            "wall_total": wall_total,
            "total": wall_total,
            "audio_duration_sec": audio_duration,
            "rtf": wall_total / audio_duration if audio_duration > 0 else 0,
        }
        return VCResult(wav=torch.from_numpy(wav), sample_rate=self.sr, timings=timings)

    def _convert_from_tokens(
        self,
        speech_tokens: np.ndarray,
        speech_token_lens: np.ndarray,
        target_voice: VoiceConditionTensors,
    ):
        token_out = self._runner(GRAPH_TOKEN_TO_MU).run(
            {
                "prompt_token": target_voice.prompt_token.astype(np.int32),
                "prompt_token_len": target_voice.prompt_token_len.astype(np.int32),
                "speech_token": speech_tokens.astype(np.int32),
                "speech_token_len": speech_token_lens.astype(np.int32),
                "embedding": target_voice.embedding.astype(np.float32),
            }
        )

        mu = token_out["mu"].astype(np.float32)
        mask = token_out["mask"].astype(np.float32)
        spks = token_out["spks"].astype(np.float32)
        prompt_mels = int(token_out["prompt_mel_len"].max())
        output_mels = int(token_out["output_mel_len"].max())

        cond = np.zeros_like(mu, dtype=np.float32)
        cond[:, :, :prompt_mels] = target_voice.prompt_feat[
            :, :prompt_mels, :
        ].transpose(0, 2, 1)
        noise = np.random.randn(*mu.shape).astype(np.float32)

        flow_out = self._runner(GRAPH_FLOW_DECODER_MEANFLOW2).run(
            {
                "noise": noise,
                "mask": mask,
                "mu": mu,
                "spks": spks,
                "cond": cond,
            }
        )
        mel = flow_out["mel"][:, :, prompt_mels : prompt_mels + output_mels].astype(
            np.float32
        )

        source_phase = np.zeros((mel.shape[0], 9, 1), dtype=np.float32)
        source_noise = np.random.randn(
            mel.shape[0], 9, mel.shape[2] * self.source_hop
        ).astype(np.float32)

        vocoder_out = self._runner(GRAPH_VOCODER_HIFT).run(
            {
                "speech_feat": mel,
                "source_phase": source_phase,
                "source_noise": source_noise,
            }
        )
        wav = vocoder_out["wav"].astype(np.float32)
        source = vocoder_out["source"].astype(np.float32)
        apply_initial_trim_fade(wav, self.trim_fade_len)
        return wav, source

    def close(self) -> None:
        for runner in list(self.runners.values()):
            runner.close()
        self.runners.clear()

        if self.activation_pool is not None:
            self.activation_pool.close()
            self.activation_pool = None

    def __enter__(self) -> "TensorRTVCBackend":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
