import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ...audio import DEC_COND_LEN, S3_SR, S3GEN_SR, load_audio_mono
from ...vc.errors import VoiceConditioningError
from ...vc.types import VCResult
from .sessions import OnnxSessions
from .solver import meanflow_euler
from .tensors import bucket_length, pad_tokens


@dataclass
class OnnxVCBackend:
    sessions: OnnxSessions
    buckets: tuple[int, ...] = (384, 512, 768, 1024)
    sr: int = S3GEN_SR

    def __post_init__(self):
        self._target_voice_cache_key = None
        self._target_voice_cache = None

    def set_target_voice(self, wav_fpath: str | Path) -> None:

        wav_fpath = Path(wav_fpath).expanduser().resolve(strict=False)
        wav_stat = wav_fpath.stat()
        cache_key = (str(wav_fpath), wav_stat.st_mtime_ns, wav_stat.st_size)

        if self._target_voice_cache_key == cache_key:
            return

        ref_wav_24k = (
            load_audio_mono(wav_fpath, S3GEN_SR, "cpu", max_len=DEC_COND_LEN)
            .unsqueeze(0)
            .numpy()
        )
        ref_wav_16k = load_audio_mono(wav_fpath, S3_SR, "cpu").unsqueeze(0).numpy()

        prompt_feat, prompt_feat_len = None, None
        if self.sessions.reference_mel_24k is not None:
            prompt_feat, prompt_feat_len = self.sessions.reference_mel_24k.run(
                None, {"wav_24k": ref_wav_24k}
            )
        else:
            import torch

            from ...models.s3gen.utils.mel import mel_spectrogram

            with torch.inference_mode():
                feat = mel_spectrogram(torch.from_numpy(ref_wav_24k)).transpose(1, 2)
                prompt_feat = feat.numpy()
                prompt_feat_len = np.array([prompt_feat.shape[1]], dtype=np.int64)

        if self.sessions.speaker_encoder is not None:
            import torch

            from ...audio.fbank import extract_fbank_features

            with torch.inference_mode():
                fbank, _, _ = extract_fbank_features(torch.from_numpy(ref_wav_16k))
                embedding = self.sessions.speaker_encoder.run(
                    None,
                    {
                        "fbank": fbank.numpy(),
                        "fbank_lengths": np.array([fbank.shape[1]], dtype=np.int64),
                    },
                )[0]
        else:
            import torch

            from ...audio.fbank import extract_fbank_features
            from ...models.s3gen.xvector import CAMPPlus

            with torch.inference_mode():
                fbank, _, _ = extract_fbank_features(torch.from_numpy(ref_wav_16k))
                spk_enc = CAMPPlus(memory_efficient=False).eval()
                embedding = spk_enc(fbank).numpy()

        if self.sessions.s3_tokenizer_quantizer is not None:
            import torch

            from ...models.s3tokenizer.s3tokenizer import S3Tokenizer

            with torch.inference_mode():
                tokenizer = S3Tokenizer()
                log_mel = tokenizer.log_mel_spectrogram(
                    torch.from_numpy(ref_wav_16k)
                ).numpy()
                mel_lens = np.array([log_mel.shape[-1]], dtype=np.int64)
                prompt_token, prompt_token_len = (
                    self.sessions.s3_tokenizer_quantizer.run(
                        None, {"log_mel": log_mel, "mel_lengths": mel_lens}
                    )
                )
        else:
            import torch

            from ...models.s3tokenizer.s3tokenizer import S3Tokenizer

            with torch.inference_mode():
                tokenizer = S3Tokenizer().eval()
                prompt_token, prompt_token_len = tokenizer(
                    torch.from_numpy(ref_wav_16k)
                )
                prompt_token = prompt_token.numpy()
                prompt_token_len = prompt_token_len.numpy()

        if prompt_feat.shape[1] != 2 * prompt_token.shape[1]:
            target_len = prompt_feat.shape[1] // 2
            prompt_token = prompt_token[:, :target_len]
            prompt_token_len = np.array([target_len], dtype=np.int64)

        self._target_voice_cache = {
            "prompt_token": prompt_token,
            "prompt_token_len": prompt_token_len,
            "prompt_feat": prompt_feat,
            "embedding": embedding,
        }
        self._target_voice_cache_key = cache_key

    def generate(
        self,
        audio,
        target_voice_path: str | Path | None = None,
        profile: bool = False,
        upscale: bool = False,
    ) -> VCResult:
        wall_start = time.perf_counter()

        if target_voice_path:
            self.set_target_voice(target_voice_path)
        elif self._target_voice_cache is None:
            raise VoiceConditioningError("Target voice is not set.")

        import torch

        from ...models.s3tokenizer.s3tokenizer import S3Tokenizer

        audio_16k = load_audio_mono(audio, S3_SR, "cpu").unsqueeze(0).numpy()

        if self.sessions.s3_tokenizer_quantizer is not None:
            with torch.inference_mode():
                tokenizer = S3Tokenizer()
                log_mel = tokenizer.log_mel_spectrogram(
                    torch.from_numpy(audio_16k)
                ).numpy()
                mel_lens = np.array([log_mel.shape[-1]], dtype=np.int64)
                speech_tokens, speech_token_lens = (
                    self.sessions.s3_tokenizer_quantizer.run(
                        None, {"log_mel": log_mel, "mel_lengths": mel_lens}
                    )
                )
        else:
            with torch.inference_mode():
                tokenizer = S3Tokenizer().eval()
                speech_tokens, speech_token_lens = tokenizer(
                    torch.from_numpy(audio_16k)
                )
                speech_tokens = speech_tokens.numpy()
                speech_token_lens = speech_token_lens.numpy()

        cache = self._target_voice_cache
        wav, _ = self.convert_from_tokens(
            speech_tokens=speech_tokens,
            speech_token_lens=speech_token_lens,
            prompt_token=cache["prompt_token"],
            prompt_token_len=cache["prompt_token_len"],
            prompt_feat=cache["prompt_feat"],
            embedding=cache["embedding"],
        )

        if upscale:
            raise NotImplementedError(
                "FlowHigh upscaling is not yet supported in ONNX backend."
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

    def convert_from_tokens(
        self,
        speech_tokens: np.ndarray,
        speech_token_lens: np.ndarray,
        prompt_token: np.ndarray,
        prompt_token_len: np.ndarray,
        prompt_feat: np.ndarray,
        embedding: np.ndarray,
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

        mu, mask, spks, prompt_mel_len, output_mel_len = self.sessions.token_to_mu.run(
            None,
            {
                "prompt_token": prompt_token.astype(np.int64),
                "prompt_token_len": prompt_token_len.astype(np.int64),
                "speech_token": speech_tokens_padded.astype(np.int64),
                "speech_token_len": speech_token_lens.astype(np.int64),
                "embedding": embedding.astype(np.float32),
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
