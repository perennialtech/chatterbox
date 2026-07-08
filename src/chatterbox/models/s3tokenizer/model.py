from typing import Tuple

import numpy as np
import torch

from .architecture import ModelConfig, S3TokenizerV2, pad_mel_batch
from .features import S3TokenizerLogMel


class S3Tokenizer(S3TokenizerV2):
    def __init__(
        self,
        name: str = "speech_tokenizer_v2_25hz",
        config: ModelConfig | None = None,
    ):
        super().__init__(name, config=config)
        self.feature_extractor = S3TokenizerLogMel(
            n_fft=400,
            n_mels=self.config.n_mels,
        )

    @staticmethod
    def _prepare_one_waveform(wav) -> torch.Tensor:
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav)
        if not torch.is_tensor(wav):
            raise TypeError(f"waveform must be a tensor, got {type(wav).__name__}")

        if wav.ndim == 1:
            return wav.unsqueeze(0)
        if wav.ndim == 2 and wav.size(0) == 1:
            return wav
        raise ValueError(
            f"each waveform must have shape [T] or [1, T], got {tuple(wav.shape)}"
        )

    def _prepare_audio(self, wavs):
        if isinstance(wavs, np.ndarray):
            wavs = torch.from_numpy(wavs)

        if torch.is_tensor(wavs):
            if wavs.ndim == 0:
                raise ValueError("audio tensor must have shape [T] or [B, T]")
            if wavs.ndim == 1:
                return [wavs.unsqueeze(0)]
            if wavs.ndim == 2:
                return [wav.unsqueeze(0) for wav in wavs]
            raise ValueError(
                f"audio tensor must have shape [T] or [B, T], got {tuple(wavs.shape)}"
            )

        return [self._prepare_one_waveform(wav) for wav in wavs]

    @torch.inference_mode()
    def forward(
        self,
        wavs: torch.Tensor,
        max_len: int = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        processed_wavs = self._prepare_audio(wavs)
        mels = []
        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)
            if max_len is not None:
                mel = mel[..., : max_len * 4]
            mels.append(mel.squeeze(0))

        mels, mel_lens = pad_mel_batch(mels)

        speech_tokens, speech_token_lens = self.quantize(mels, mel_lens.to(self.device))
        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(self, audio: torch.Tensor):
        return self.feature_extractor(audio)
