from __future__ import annotations

from dataclasses import dataclass

import librosa
import torch
import torch.nn.functional as F
from s3tokenizer.model_v2 import ModelConfig, S3TokenizerV2
from s3tokenizer.utils import padding

from chatterbox.types import TokenBatch

S3_SR = 16_000
S3_HOP = 160
S3_TOKEN_RATE = 25
S3_TOKEN_HOP = 640
SPEECH_VOCAB_SIZE = 6561


@dataclass(frozen=True)
class S3TokenizerConfig:
    name: str = "speech_tokenizer_v2_25hz"
    n_fft: int = 400


class SourceTokenizer(torch.nn.Module):
    def __init__(self, config: S3TokenizerConfig = S3TokenizerConfig()):
        super().__init__()
        self.config = config
        self.backend = S3TokenizerV2(config.name)
        self.n_fft = config.n_fft

        model_config = ModelConfig()
        mel_filters = librosa.filters.mel(
            sr=S3_SR,
            n_fft=self.n_fft,
            n_mels=model_config.n_mels,
        )
        self.register_buffer(
            "_mel_filters", torch.tensor(mel_filters, dtype=torch.float32)
        )
        self.register_buffer("window", torch.hann_window(self.n_fft))

    @property
    def device(self) -> torch.device:
        return next(self.backend.parameters()).device

    @torch.inference_mode()
    def encode(
        self, wav_16k: torch.Tensor, max_tokens: int | None = None
    ) -> TokenBatch:
        if wav_16k.ndim != 2:
            raise ValueError("Tokenizer input must have shape [B, samples].")

        mels = []
        for wav in wav_16k:
            mel = self.log_mel_spectrogram(wav.unsqueeze(0))
            if max_tokens is not None:
                mel = mel[..., : max_tokens * 4]
            mels.append(mel.squeeze(0))

        padded_mels, mel_lengths = padding(mels)
        tokens, lengths = self.backend.quantize(
            padded_mels.to(self.device),
            mel_lengths.to(self.device),
        )
        return TokenBatch(tokens.long().detach(), lengths.long().detach())

    def log_mel_spectrogram(
        self, audio: torch.Tensor, right_padding: int = 0
    ) -> torch.Tensor:
        audio = audio.to(self.device)
        if right_padding:
            audio = F.pad(audio, (0, right_padding))

        stft = torch.stft(
            audio,
            self.n_fft,
            S3_HOP,
            window=self.window,
            return_complex=True,
        )
        magnitudes = stft[..., :-1].abs().square()
        mel_spec = self._mel_filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        return (log_spec + 4.0) / 4.0
