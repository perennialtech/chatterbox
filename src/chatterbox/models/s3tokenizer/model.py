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

    def _prepare_audio(self, wavs):
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            processed_wavs.append(wav)
        return processed_wavs

    @torch.inference_mode()
    def forward(
        self,
        wavs: torch.Tensor,
        max_len: int = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        processed_wavs = self._prepare_audio(wavs)
        mels, mel_lens = [], []
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
