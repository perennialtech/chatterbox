from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ..constants import (flow_decoder_graph_name, token_to_mu_graph_name,
                         vocoder_graph_name)
from ..errors import OnnxRuntimeError
from .sessions import OnnxSessions

FlowNoiseCallback = Callable[[int, tuple[int, ...]], np.ndarray]


@dataclass(frozen=True)
class OnnxReferenceCondition:
    prompt_token: np.ndarray
    prompt_token_len: np.ndarray
    prompt_feat: np.ndarray
    embedding: np.ndarray


class OnnxS3Gen:
    def __init__(self, sessions: OnnxSessions):
        self.sessions = sessions
        self.manifest = sessions.manifest
        self.constants = self.manifest["constants"]

        self.vocab_size = int(self.constants["speech_vocab_size"])
        self.sil_token = int(self.constants["s3gen_sil"])
        self.token_mel_ratio = int(self.constants["token_mel_ratio"])
        self.final_context_token_count = int(
            self.constants["final_context_token_count"]
        )
        self.flow_chunk_tokens = int(self.constants["flow_chunk_tokens"])
        self.flow_context_tokens = int(self.constants["flow_context_tokens"])
        self.ref_max_prompt_tokens = int(
            self.constants["prompt_limits"]["ref_max_prompt_tokens"]
        )
        self.source_hop = int(self.constants["source_hop"])
        self.vocoder_harmonics = int(self.constants["vocoder_harmonics"])
        self.token_buckets = tuple(
            int(x) for x in self.constants["token_to_mu_token_buckets"]
        )
        self.flow_mel_buckets = tuple(
            int(x) for x in self.constants["flow_mel_buckets"]
        )
        self.vocoder_mel_buckets = tuple(
            int(x) for x in self.constants["vocoder_mel_buckets"]
        )

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: Path,
        providers: list[str] | None = None,
    ) -> "OnnxS3Gen":
        return cls(OnnxSessions.from_artifact_dir(artifact_dir, providers=providers))

    def _bucket(self, length: int, buckets: tuple[int, ...], name: str) -> int:
        for bucket in buckets:
            if length <= bucket:
                return bucket
        raise OnnxRuntimeError(
            f"{name} length {length} exceeds largest bucket {buckets[-1]}"
        )

    def _validate_tokens(self, name: str, tokens: np.ndarray) -> None:
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError(f"{name} must have shape [1, T]")
        if np.any(tokens < 0) or np.any(tokens >= self.vocab_size):
            raise ValueError(
                f"{name} contains token IDs outside [0, {self.vocab_size})"
            )

    def drop_invalid_tokens(self, speech_tokens) -> np.ndarray:
        tokens = np.asarray(speech_tokens)
        if tokens.ndim == 2 and tokens.shape[0] == 1:
            tokens = tokens[0]
        if tokens.ndim != 1:
            raise ValueError("speech_tokens must have shape [T] or [1, T]")
        tokens = tokens.astype(np.int64, copy=False)
        tokens = tokens[(tokens >= 0) & (tokens < self.vocab_size)]
        if tokens.size == 0:
            raise ValueError("At least one valid speech token is required")
        return tokens.reshape(1, -1)

    def prepare_ref_condition(
        self,
        ref_dict: Mapping[str, object] | OnnxReferenceCondition,
    ) -> OnnxReferenceCondition:
        if isinstance(ref_dict, OnnxReferenceCondition):
            data = {
                "prompt_token": ref_dict.prompt_token,
                "prompt_token_len": ref_dict.prompt_token_len,
                "prompt_feat": ref_dict.prompt_feat,
                "embedding": ref_dict.embedding,
            }
        else:
            data = dict(ref_dict)

        missing = {
            "prompt_token",
            "prompt_token_len",
            "prompt_feat",
            "embedding",
        } - set(data)
        if missing:
            raise ValueError(f"Missing reference tensors: {sorted(missing)}")

        prompt_token = np.asarray(data["prompt_token"], dtype=np.int32)
        prompt_token_len = np.asarray(data["prompt_token_len"], dtype=np.int32)
        prompt_feat = np.asarray(data["prompt_feat"], dtype=np.float32)
        embedding = np.asarray(data["embedding"], dtype=np.float32)

        if prompt_token.ndim == 1:
            prompt_token = prompt_token[None, :]
        if prompt_token_len.ndim == 0:
            prompt_token_len = prompt_token_len.reshape(1)
        if (
            prompt_feat.ndim != 3
            or prompt_feat.shape[0] != 1
            or prompt_feat.shape[2] != 80
        ):
            raise ValueError("prompt_feat must have shape [1, 2P, 80]")
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        if embedding.shape != (1, 192):
            raise ValueError("embedding must have shape [1, 192]")
        if prompt_token.ndim != 2 or prompt_token.shape[0] != 1:
            raise ValueError("prompt_token must have shape [1, P]")
        if prompt_token_len.shape != (1,):
            raise ValueError("prompt_token_len must have shape [1]")

        prompt_len = int(prompt_token_len[0])
        if prompt_len <= 0:
            raise ValueError("prompt_token_len must be positive")
        prompt_len = min(prompt_len, self.ref_max_prompt_tokens)
        if prompt_len > prompt_token.shape[1]:
            raise ValueError("prompt_token_len exceeds prompt_token width")
        if prompt_feat.shape[1] < prompt_len * self.token_mel_ratio:
            raise ValueError("prompt_feat is shorter than 2x prompt_token_len")

        prompt_token = np.ascontiguousarray(
            prompt_token[:, :prompt_len], dtype=np.int32
        )
        prompt_token_len = np.asarray([prompt_len], dtype=np.int32)
        prompt_feat = np.ascontiguousarray(
            prompt_feat[:, : prompt_len * self.token_mel_ratio, :],
            dtype=np.float32,
        )
        embedding = np.ascontiguousarray(embedding, dtype=np.float32)

        self._validate_tokens("prompt_token", prompt_token)
        return OnnxReferenceCondition(
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            embedding=embedding,
        )

    def _append_silence_context(
        self,
        speech_tokens: np.ndarray,
        context_tokens: int,
    ) -> np.ndarray:
        if context_tokens <= 0:
            return speech_tokens
        tail = np.full((1, context_tokens), self.sil_token, dtype=np.int32)
        return np.concatenate(
            [speech_tokens.astype(np.int32, copy=False), tail], axis=1
        )

    def _pad_tokens_to_bucket(
        self,
        prompt_token: np.ndarray,
        speech_tokens: np.ndarray,
    ) -> tuple[np.ndarray, int]:
        total_len = prompt_token.shape[1] + speech_tokens.shape[1]
        bucket = self._bucket(total_len, self.token_buckets, "token")
        token = np.zeros((1, bucket), dtype=np.int32)
        token[:, : prompt_token.shape[1]] = prompt_token
        token[:, prompt_token.shape[1] : total_len] = speech_tokens
        self._validate_tokens("token", token[:, :total_len])
        return token, bucket

    def _pad_mel_to_bucket(
        self,
        value: np.ndarray,
        buckets: tuple[int, ...],
        name: str,
    ) -> tuple[np.ndarray, int]:
        bucket = self._bucket(value.shape[2], buckets, name)
        if value.shape[2] == bucket:
            return np.ascontiguousarray(value, dtype=np.float32), bucket
        padded = np.zeros((value.shape[0], value.shape[1], bucket), dtype=np.float32)
        padded[:, :, : value.shape[2]] = value
        return padded, bucket

    def _run_flow_window(
        self,
        *,
        chunk_index: int,
        speech_tokens: np.ndarray,
        ref: OnnxReferenceCondition,
        flow_noise_callback: FlowNoiseCallback | None,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, dict]:
        token, token_bucket = self._pad_tokens_to_bucket(
            ref.prompt_token, speech_tokens
        )
        speech_token_len = np.asarray([speech_tokens.shape[1]], dtype=np.int32)

        token_to_mu = self.sessions.runner(token_to_mu_graph_name(token_bucket))
        mu_outputs = token_to_mu.run(
            {
                "token": token,
                "prompt_token_len": ref.prompt_token_len,
                "speech_token_len": speech_token_len,
                "embedding": ref.embedding,
            }
        )

        mu = np.ascontiguousarray(mu_outputs["mu"], dtype=np.float32)
        mask = np.ascontiguousarray(mu_outputs["mask"], dtype=np.float32)
        spks = np.ascontiguousarray(mu_outputs["spks"], dtype=np.float32)
        prompt_mel_len = int(np.asarray(mu_outputs["prompt_mel_len"])[0])
        output_mel_len = int(np.asarray(mu_outputs["output_mel_len"])[0])

        cond = np.zeros_like(mu, dtype=np.float32)
        cond[:, :, :prompt_mel_len] = ref.prompt_feat.transpose(0, 2, 1)

        mu, flow_mel_bucket = self._pad_mel_to_bucket(
            mu, self.flow_mel_buckets, "flow mel"
        )
        mask, _ = self._pad_mel_to_bucket(mask, self.flow_mel_buckets, "flow mask")
        cond, _ = self._pad_mel_to_bucket(cond, self.flow_mel_buckets, "flow cond")

        noise_shape = mu.shape
        if flow_noise_callback is None:
            noise = rng.standard_normal(noise_shape).astype(np.float32)
        else:
            noise = np.asarray(
                flow_noise_callback(chunk_index, noise_shape), dtype=np.float32
            )
            if noise.shape != noise_shape:
                raise ValueError(
                    f"flow noise callback returned {noise.shape}, expected {noise_shape}"
                )

        flow = self.sessions.runner(flow_decoder_graph_name(flow_mel_bucket))
        flow_mel = flow.run(
            {
                "noise": noise,
                "mask": mask,
                "mu": mu,
                "spks": spks,
                "cond": cond,
            }
        )["mel"]

        window_mel = np.ascontiguousarray(
            flow_mel[:, :, prompt_mel_len : prompt_mel_len + output_mel_len],
            dtype=np.float32,
        )
        return window_mel, {
            "chunk_index": chunk_index,
            "token_bucket": token_bucket,
            "flow_mel_bucket": flow_mel_bucket,
            "prompt_mel_len": prompt_mel_len,
            "output_mel_len": output_mel_len,
            "sliced_after_prompt_mel_len": prompt_mel_len,
        }

    def flow_inference(
        self,
        speech_tokens,
        ref_dict: Mapping[str, object] | OnnxReferenceCondition,
        *,
        drop_invalid_tokens: bool = True,
        flow_noise_callback: FlowNoiseCallback | None = None,
        rng: np.random.Generator | None = None,
        return_debug: bool = False,
    ):
        rng = np.random.default_rng() if rng is None else rng
        ref = self.prepare_ref_condition(ref_dict)

        if drop_invalid_tokens:
            speech_tokens = self.drop_invalid_tokens(speech_tokens)
        else:
            speech_tokens = np.asarray(speech_tokens, dtype=np.int32)
            if speech_tokens.ndim == 1:
                speech_tokens = speech_tokens[None, :]
            self._validate_tokens("speech_tokens", speech_tokens)

        target_token_len = int(speech_tokens.shape[1])
        original_mel_len = target_token_len * self.token_mel_ratio
        chunk_tokens = max(1, self.flow_chunk_tokens)
        context_tokens = max(self.flow_context_tokens, self.final_context_token_count)

        chunks = []
        debug_chunks = []
        for chunk_index, center_start in enumerate(
            range(0, target_token_len, chunk_tokens)
        ):
            center_end = min(center_start + chunk_tokens, target_token_len)
            left_start = max(0, center_start - context_tokens)
            right_end = min(target_token_len, center_end + context_tokens)

            window = np.ascontiguousarray(
                speech_tokens[:, left_start:right_end],
                dtype=np.int32,
            )
            appended_silence_tokens = 0
            if center_end == target_token_len:
                appended_silence_tokens = self.final_context_token_count
                window = self._append_silence_context(window, appended_silence_tokens)

            window_mel, chunk_debug = self._run_flow_window(
                chunk_index=chunk_index,
                speech_tokens=window,
                ref=ref,
                flow_noise_callback=flow_noise_callback,
                rng=rng,
            )

            mel_start = (center_start - left_start) * self.token_mel_ratio
            mel_end = mel_start + (center_end - center_start) * self.token_mel_ratio
            chunk = np.ascontiguousarray(
                window_mel[:, :, mel_start:mel_end], dtype=np.float32
            )
            chunks.append(chunk)

            chunk_debug.update(
                {
                    "center_start": center_start,
                    "center_end": center_end,
                    "left_start": left_start,
                    "right_end": right_end,
                    "mel_start": mel_start,
                    "mel_end": mel_end,
                    "appended_silence_tokens": appended_silence_tokens,
                    "chunk_mel_len": int(chunk.shape[2]),
                }
            )
            debug_chunks.append(chunk_debug)

        output_mels = np.ascontiguousarray(
            np.concatenate(chunks, axis=2), dtype=np.float32
        )
        output_mels = output_mels[:, :, :original_mel_len]
        if output_mels.shape[2] != original_mel_len:
            raise OnnxRuntimeError(
                f"concatenated mel length {output_mels.shape[2]} != expected {original_mel_len}"
            )

        debug = {
            "original_token_len": target_token_len,
            "original_mel_len": original_mel_len,
            "concat_mel_len": int(output_mels.shape[2]),
            "chunks": debug_chunks,
        }
        return (output_mels, debug) if return_debug else output_mels

    def _make_source_phase(
        self,
        rng: np.random.Generator,
    ) -> np.ndarray:
        phase = rng.uniform(
            -math.pi,
            math.pi,
            size=(1, self.vocoder_harmonics, 1),
        ).astype(np.float32)
        phase[:, :1, :] = 0.0
        return phase

    def _apply_output_fades(self, output_wavs: np.ndarray) -> np.ndarray:
        if output_wavs.shape[1] == 0:
            return output_wavs

        n_trim = int(self.constants["fade"]["n_trim"])
        trim_fade = np.zeros(2 * n_trim, dtype=np.float32)
        trim_fade[n_trim:] = (np.cos(np.linspace(math.pi, 0.0, n_trim)) + 1.0) / 2.0
        end_fade = (np.cos(np.linspace(0.0, math.pi, 2 * n_trim)) + 1.0) / 2.0

        fade_len = min(trim_fade.size, end_fade.size, output_wavs.shape[1] // 4)
        if fade_len <= 0:
            return output_wavs

        output_wavs[:, :fade_len] *= trim_fade[-fade_len:]
        output_wavs[:, -fade_len:] *= end_fade[:fade_len]
        return output_wavs

    def vocoder_inference(
        self,
        speech_feat: np.ndarray,
        *,
        original_mel_len: int | None = None,
        source_phase: np.ndarray | None = None,
        source_noise: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng() if rng is None else rng
        speech_feat = np.asarray(speech_feat, dtype=np.float32)
        if (
            speech_feat.ndim != 3
            or speech_feat.shape[0] != 1
            or speech_feat.shape[1] != 80
        ):
            raise ValueError("speech_feat must have shape [1, 80, T]")

        original_mel_len = (
            speech_feat.shape[2] if original_mel_len is None else int(original_mel_len)
        )
        padded_feat, mel_bucket = self._pad_mel_to_bucket(
            speech_feat,
            self.vocoder_mel_buckets,
            "vocoder mel",
        )

        if source_phase is None:
            source_phase = self._make_source_phase(rng)
        else:
            source_phase = np.asarray(source_phase, dtype=np.float32)
        if source_phase.shape != (1, self.vocoder_harmonics, 1):
            raise ValueError(
                f"source_phase must have shape [1, {self.vocoder_harmonics}, 1]"
            )

        source_noise_shape = (
            1,
            self.vocoder_harmonics,
            mel_bucket * self.source_hop,
        )
        if source_noise is None:
            source_noise = rng.standard_normal(source_noise_shape).astype(np.float32)
        else:
            source_noise = np.asarray(source_noise, dtype=np.float32)
        if source_noise.shape != source_noise_shape:
            raise ValueError(
                f"source_noise must have shape {source_noise_shape}, got {source_noise.shape}"
            )

        vocoder = self.sessions.runner(vocoder_graph_name(mel_bucket))
        outputs = vocoder.run(
            {
                "speech_feat": padded_feat,
                "source_phase": np.ascontiguousarray(source_phase, dtype=np.float32),
                "source_noise": np.ascontiguousarray(source_noise, dtype=np.float32),
            }
        )
        wav = np.ascontiguousarray(outputs["wav"], dtype=np.float32)
        source = np.ascontiguousarray(outputs["source"], dtype=np.float32)

        original_samples = min(wav.shape[1], original_mel_len * self.source_hop)
        wav = np.ascontiguousarray(wav[:, :original_samples], dtype=np.float32)
        source = np.ascontiguousarray(source[:, :, :original_samples], dtype=np.float32)
        wav = self._apply_output_fades(wav)
        return wav, source

    def inference(
        self,
        speech_tokens,
        ref_dict: Mapping[str, object] | OnnxReferenceCondition,
        *,
        drop_invalid_tokens: bool = True,
        flow_noise_callback: FlowNoiseCallback | None = None,
        source_phase: np.ndarray | None = None,
        source_noise: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
        return_mel: bool = False,
        return_debug: bool = False,
    ):
        rng = np.random.default_rng() if rng is None else rng
        flow_result = self.flow_inference(
            speech_tokens,
            ref_dict,
            drop_invalid_tokens=drop_invalid_tokens,
            flow_noise_callback=flow_noise_callback,
            rng=rng,
            return_debug=return_debug,
        )
        if return_debug:
            output_mels, debug = flow_result
        else:
            output_mels = flow_result
            debug = None

        wav, source = self.vocoder_inference(
            output_mels,
            original_mel_len=output_mels.shape[2],
            source_phase=source_phase,
            source_noise=source_noise,
            rng=rng,
        )

        outputs = [wav, source]
        if return_mel:
            outputs.append(output_mels)
        if return_debug:
            outputs.append(debug)
        return tuple(outputs)
