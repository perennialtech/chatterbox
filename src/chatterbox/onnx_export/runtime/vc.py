from dataclasses import dataclass

import numpy as np

from ...audio import S3GEN_SR
from .sessions import OnnxSessions
from .solver import meanflow_euler
from .tensors import bucket_length, pad_tokens


@dataclass
class OnnxVCBackend:
    sessions: OnnxSessions
    buckets: tuple[int, ...] = (384, 512, 768, 1024)

    def convert_from_tokens(
        self,
        speech_tokens: np.ndarray,
        speech_token_lens: np.ndarray,
        prompt_token: np.ndarray,
        prompt_token_len: np.ndarray,
        prompt_feat: np.ndarray,
        spks: np.ndarray,
        noise: np.ndarray | None = None,
        source_phase: np.ndarray | None = None,
        source_noise: np.ndarray | None = None,
    ):
        if self.sessions.token_to_mu is None:
            raise RuntimeError("token_to_mu ONNX session is required")
        if self.sessions.vocoder is None:
            raise RuntimeError("vocoder ONNX session is required")

        prompt_len = int(prompt_token_len.max())
        speech_len = int(speech_token_lens.max())
        token_bucket = bucket_length(prompt_len + speech_len, self.buckets)
        speech_bucket = max(1, token_bucket - prompt_token.shape[1])
        speech_tokens_padded = pad_tokens(speech_tokens, speech_bucket)

        mu, mask, prompt_mel_len, output_mel_len = self.sessions.token_to_mu.run(
            None,
            {
                "prompt_token": prompt_token.astype(np.int64),
                "prompt_token_len": prompt_token_len.astype(np.int64),
                "speech_token": speech_tokens_padded.astype(np.int64),
                "speech_token_len": speech_token_lens.astype(np.int64),
            },
        )

        if noise is None:
            noise = np.random.randn(*mu.shape).astype(mu.dtype)

        cond = np.zeros_like(mu)
        prompt_mels = int(prompt_mel_len.max())
        cond[:, :, :prompt_mels] = prompt_feat[:, :prompt_mels, :].transpose(0, 2, 1)

        if self.sessions.flow_decoder is not None:
            (mel,) = self.sessions.flow_decoder.run(
                None,
                {
                    "noise": noise,
                    "mask": mask,
                    "mu": mu,
                    "spks": spks,
                    "cond": cond,
                },
            )
        else:
            if self.sessions.conditional_decoder_step is None:
                raise RuntimeError("conditional_decoder_step ONNX session is required")
            mel = meanflow_euler(
                self.sessions.conditional_decoder_step,
                noise,
                mu,
                mask,
                spks,
                cond,
                np.asarray([0.0, 0.5, 1.0], dtype=mu.dtype),
            )

        output_mels = int(output_mel_len.max())
        mel = mel[:, :, prompt_mels : prompt_mels + output_mels]

        if source_phase is None:
            source_phase = np.zeros((mel.shape[0], 9, 1), dtype=mel.dtype)
        if source_noise is None:
            source_noise = np.random.randn(mel.shape[0], 9, mel.shape[2] * 120).astype(
                mel.dtype
            )

        wav, source = self.sessions.vocoder.run(
            None,
            {
                "speech_feat": mel.astype(np.float32),
                "source_phase": source_phase.astype(np.float32),
                "source_noise": source_noise.astype(np.float32),
            },
        )

        trim_len = S3GEN_SR // 25
        fade = np.zeros(trim_len, dtype=wav.dtype)
        half = trim_len // 2
        fade[half:] = (np.cos(np.linspace(np.pi, 0, trim_len - half)) + 1) / 2
        wav[:, :trim_len] *= fade
        return wav, source
