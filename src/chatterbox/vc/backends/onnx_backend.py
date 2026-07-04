from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch

from ...audio import DEC_COND_LEN, S3GEN_SR
from ...onnx_export.constants import (GRAPH_FLOW_DECODER_MEANFLOW2,
                                      GRAPH_REFERENCE_MEL_24K,
                                      GRAPH_S3_TOKENIZER_QUANTIZER,
                                      GRAPH_SPEAKER_ENCODER, GRAPH_TOKEN_TO_MU,
                                      GRAPH_VOCODER_HIFT)
from ...onnx_export.runtime.sessions import OnnxSessions
from ..conditioning import VoiceConditionTensors
from ..errors import BackendUnavailableError, VoiceConditioningError
from ..preprocess import (compute_fbank, compute_s3_log_mel, load_wav_16k,
                          load_wav_24k)
from ..types import VCResult


def _trim_fade(length: int, dtype) -> np.ndarray:
    half = length // 2
    fade = np.zeros(length, dtype=dtype)
    fade[half:] = (np.cos(np.linspace(np.pi, 0, length - half)) + 1) / 2
    return fade


@dataclass
class OnnxVCBackend:
    sessions: OnnxSessions
    sr: int = S3GEN_SR

    def __post_init__(self):
        self._target_voice_cache: VoiceConditionTensors | None = None
        self.runners = {
            name: self.sessions.runner(name)
            for name in (
                GRAPH_REFERENCE_MEL_24K,
                GRAPH_SPEAKER_ENCODER,
                GRAPH_S3_TOKENIZER_QUANTIZER,
                GRAPH_TOKEN_TO_MU,
                GRAPH_FLOW_DECODER_MEANFLOW2,
                GRAPH_VOCODER_HIFT,
            )
        }
        self.constants = self.sessions.manifest["constants"]
        self.source_hop = int(self.constants["source_hop"])
        self.trim_fade_len = int(self.constants["trim_fade_len"])

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        precision: Literal["fp32", "fp16"] = "fp32",
        providers: list[str] | None = None,
    ) -> "OnnxVCBackend":
        sessions = OnnxSessions.from_artifact_dir(
            Path(artifact_dir), precision=precision, providers=providers
        )
        return cls(sessions=sessions)

    def set_target_voice_from_tensors(
        self, target_voice: dict | VoiceConditionTensors
    ) -> None:
        self._target_voice_cache = VoiceConditionTensors.from_mapping(target_voice)

    def convert_from_path(
        self,
        audio_path: str | Path,
        target_voice_path: str | Path | None = None,
        profile: bool = False,
        upscale: bool = False,
    ) -> VCResult:
        if target_voice_path:
            ref_wav_24k = load_wav_24k(target_voice_path, "cpu", max_len=DEC_COND_LEN)
            ref_wav_16k = load_wav_16k(target_voice_path, "cpu")
            self.set_target_voice_from_tensors(
                self._extract_target_voice_tensors(
                    ref_wav_24k.numpy(), ref_wav_16k.numpy()
                )
            )

        audio_16k = load_wav_16k(audio_path, "cpu")
        return self.convert_from_tensors(
            audio_16k, self._target_voice_cache, profile, upscale
        )

    def _require_runner(self, name: str):
        if name not in self.runners:
            raise BackendUnavailableError(f"Required ONNX graph {name} is missing")
        return self.runners[name]

    def _tokenize_audio(self, audio_16k: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        log_mel, mel_lengths = compute_s3_log_mel(audio_16k)
        out = self._require_runner(GRAPH_S3_TOKENIZER_QUANTIZER).run(
            {"log_mel": log_mel, "mel_lengths": mel_lengths}
        )
        return out["speech_tokens"].astype(np.int64), out[
            "speech_token_lengths"
        ].astype(np.int64)

    def _extract_target_voice_tensors(
        self, ref_wav_24k: np.ndarray, ref_wav_16k: np.ndarray
    ) -> VoiceConditionTensors:
        mel_out = self._require_runner(GRAPH_REFERENCE_MEL_24K).run(
            {"wav_24k": np.ascontiguousarray(ref_wav_24k.astype(np.float32))}
        )
        prompt_feat = mel_out["prompt_feat"].astype(np.float32)
        prompt_feat_len = mel_out["prompt_feat_len"].astype(np.int64)

        fbank, fbank_lengths = compute_fbank(torch.from_numpy(ref_wav_16k))
        speaker_out = self._require_runner(GRAPH_SPEAKER_ENCODER).run(
            {"fbank": fbank, "fbank_lengths": fbank_lengths}
        )
        embedding = speaker_out["embedding"].astype(np.float32)

        prompt_token, prompt_token_len = self._tokenize_audio(
            torch.from_numpy(ref_wav_16k)
        )

        if prompt_feat.shape[1] != 2 * prompt_token.shape[1]:
            target_len = prompt_feat.shape[1] // 2
            prompt_token = prompt_token[:, :target_len]
            prompt_token_len = np.array([target_len], dtype=np.int64)

        return VoiceConditionTensors.from_mapping(
            {
                "prompt_token": prompt_token,
                "prompt_token_len": prompt_token_len,
                "prompt_feat": prompt_feat,
                "prompt_feat_len": prompt_feat_len,
                "embedding": embedding,
            }
        )

    def convert_from_tensors(
        self,
        audio_16k: torch.Tensor,
        target_voice: dict | VoiceConditionTensors | None = None,
        profile: bool = False,
        upscale: bool = False,
    ) -> VCResult:
        if upscale:
            raise BackendUnavailableError(
                "FlowHigh upscaling is not supported by the ONNX backend."
            )

        wall_start = time.perf_counter()

        if target_voice is None:
            target_voice = self._target_voice_cache
        if target_voice is None:
            raise VoiceConditioningError("Target voice is not set.")

        condition = VoiceConditionTensors.from_mapping(target_voice)
        speech_tokens, speech_token_lens = self._tokenize_audio(audio_16k)

        wav, _ = self._convert_from_tokens(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            target_voice=condition,
        )

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
        noise: np.ndarray | None = None,
        source_phase: np.ndarray | None = None,
        source_noise: np.ndarray | None = None,
    ):
        token_out = self._require_runner(GRAPH_TOKEN_TO_MU).run(
            {
                "prompt_token": target_voice.prompt_token,
                "prompt_token_len": target_voice.prompt_token_len,
                "speech_token": speech_tokens.astype(np.int64),
                "speech_token_len": speech_token_lens.astype(np.int64),
                "embedding": target_voice.embedding.astype(np.float32),
            }
        )

        mu = token_out["mu"].astype(np.float32)
        mask = token_out["mask"].astype(np.float32)
        spks = token_out["spks"].astype(np.float32)
        prompt_mel_len = token_out["prompt_mel_len"].astype(np.int64)
        output_mel_len = token_out["output_mel_len"].astype(np.int64)

        prompt_mels = int(prompt_mel_len.max())
        output_mels = int(output_mel_len.max())

        cond = np.zeros_like(mu, dtype=np.float32)
        cond[:, :, :prompt_mels] = target_voice.prompt_feat[
            :, :prompt_mels, :
        ].transpose(0, 2, 1)

        if noise is None:
            noise = np.random.randn(*mu.shape).astype(np.float32)

        flow_out = self._require_runner(GRAPH_FLOW_DECODER_MEANFLOW2).run(
            {
                "noise": noise.astype(np.float32),
                "mask": mask,
                "mu": mu,
                "spks": spks,
                "cond": cond,
            }
        )
        mel = flow_out["mel"][:, :, prompt_mels : prompt_mels + output_mels].astype(
            np.float32
        )

        if source_phase is None:
            source_phase = np.zeros((mel.shape[0], 9, 1), dtype=np.float32)
        if source_noise is None:
            source_noise = np.random.randn(
                mel.shape[0], 9, mel.shape[2] * self.source_hop
            ).astype(np.float32)

        vocoder_out = self._require_runner(GRAPH_VOCODER_HIFT).run(
            {
                "speech_feat": mel,
                "source_phase": source_phase.astype(np.float32),
                "source_noise": source_noise.astype(np.float32),
            }
        )

        wav = vocoder_out["wav"].astype(np.float32)
        source = vocoder_out["source"].astype(np.float32)
        wav[:, : self.trim_fade_len] *= _trim_fade(self.trim_fade_len, wav.dtype)
        return wav, source
