from typing import List, Tuple

import numpy as np
import torch
from s3tokenizer.model_v2 import ModelConfig, S3TokenizerV2
from s3tokenizer.utils import padding

from ...audio.constants import S3_TOKEN_RATE
from .features import S3TokenizerLogMel


class S3Tokenizer(S3TokenizerV2):
    ignore_state_dict_missing = (
        "feature_extractor._mel_filters",
        "feature_extractor.window",
    )

    def __init__(
        self,
        name: str = "speech_tokenizer_v2_25hz",
        config: ModelConfig = ModelConfig(),
    ):
        super().__init__(name)
        self.feature_extractor = S3TokenizerLogMel(n_fft=400, n_mels=config.n_mels)

    def pad(self, wavs, sr) -> List[torch.Tensor]:
        processed_wavs = []
        for wav in wavs:
            if isinstance(wav, np.ndarray):
                wav = torch.from_numpy(wav)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)

            n_tokens = (wav.shape[1] / sr) * S3_TOKEN_RATE
            n_tokens = np.ceil(n_tokens)
            intended_wav_len = n_tokens * (sr / S3_TOKEN_RATE)
            intended_wav_len = int(intended_wav_len)
            wav = torch.nn.functional.pad(
                wav, (0, intended_wav_len - wav.shape[-1]), mode="constant", value=0
            )
            processed_wavs.append(wav)
        return processed_wavs

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
        accelerator: "Accelerator" = None,
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

        mels, mel_lens = padding(mels)
        if accelerator is None:
            tokenizer = self
        else:
            tokenizer = accelerator.unwrap_model(self)

        speech_tokens, speech_token_lens = tokenizer.quantize(
            mels, mel_lens.to(self.device)
        )
        return (
            speech_tokens.long().detach(),
            speech_token_lens.long().detach(),
        )

    def log_mel_spectrogram(self, audio: torch.Tensor, padding: int = 0):
        return self.feature_extractor(audio, padding=padding)
