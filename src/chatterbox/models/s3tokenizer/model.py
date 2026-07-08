from typing import Tuple

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from ...audio.constants import S3_HOP, S3_SR
from .architecture import ModelConfig, S3TokenizerV2
from .features import S3TokenizerLogMel

TOKENIZER_CHUNK_SECONDS = 30.0
TOKENIZER_OVERLAP_SECONDS = 2.0


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

    @property
    def _mel_frames_per_second(self) -> float:
        return S3_SR / S3_HOP

    @property
    def _tokenizer_chunk_frames(self) -> int:
        return int(round(TOKENIZER_CHUNK_SECONDS * self._mel_frames_per_second))

    @property
    def _tokenizer_overlap_frames(self) -> int:
        return int(round(TOKENIZER_OVERLAP_SECONDS * self._mel_frames_per_second))

    def _quantize_mel_full(self, mel: torch.Tensor) -> tuple[torch.Tensor, int]:
        mel_len = torch.tensor([mel.size(-1)], dtype=torch.long, device=self.device)
        speech_tokens, speech_token_lens = self.quantize(mel, mel_len)
        token_len = int(speech_token_lens[0].detach().cpu())
        return speech_tokens[0, :token_len].long(), token_len

    def _quantize_mel_chunked(self, mel: torch.Tensor) -> tuple[torch.Tensor, int]:
        chunk_frames = self._tokenizer_chunk_frames
        overlap_frames = self._tokenizer_overlap_frames
        if mel.size(-1) <= chunk_frames:
            return self._quantize_mel_full(mel)

        if overlap_frames <= 0 or overlap_frames >= chunk_frames:
            raise ValueError(
                "tokenizer overlap must be positive and smaller than chunk size"
            )

        step_frames = chunk_frames - overlap_frames
        half_overlap = overlap_frames // 2
        mel_to_token = 4.0

        pieces = []
        start = 0
        while start < mel.size(-1):
            end = min(start + chunk_frames, mel.size(-1))
            chunk = mel[..., start:end].contiguous()
            chunk_tokens, chunk_token_len = self._quantize_mel_full(chunk)

            keep_mel_start = 0 if start == 0 else half_overlap
            keep_mel_end = (
                end - start if end == mel.size(-1) else end - start - half_overlap
            )
            keep_mel_end = max(keep_mel_start, keep_mel_end)

            keep_token_start = min(
                chunk_token_len, int(round(keep_mel_start / mel_to_token))
            )
            keep_token_end = min(
                chunk_token_len, int(round(keep_mel_end / mel_to_token))
            )
            if keep_token_end > keep_token_start:
                pieces.append(chunk_tokens[keep_token_start:keep_token_end])

            if end == mel.size(-1):
                break
            start += step_frames

        if not pieces:
            return self._quantize_mel_full(mel)

        speech_tokens = torch.cat(pieces, dim=0).long()
        return speech_tokens, int(speech_tokens.numel())

    def _quantize_mel(self, mel: torch.Tensor) -> tuple[torch.Tensor, int]:
        mel = mel.to(self.device)
        if mel.size(-1) > self._tokenizer_chunk_frames:
            return self._quantize_mel_chunked(mel)
        return self._quantize_mel_full(mel)

    @torch.inference_mode()
    def forward(
        self,
        wavs: torch.Tensor,
        max_len: int = None,
    ) -> Tuple[torch.Tensor, torch.LongTensor]:
        processed_wavs = self._prepare_audio(wavs)
        token_sequences = []
        token_lengths = []

        for wav in processed_wavs:
            wav = wav.to(self.device)
            mel = self.log_mel_spectrogram(wav)
            if max_len is not None:
                mel = mel[..., : max_len * 4]
            tokens, token_len = self._quantize_mel(mel)
            token_sequences.append(tokens.detach())
            token_lengths.append(token_len)

        speech_tokens = pad_sequence(
            token_sequences,
            batch_first=True,
            padding_value=0,
        ).to(device=self.device, dtype=torch.long)
        speech_token_lens = torch.tensor(
            token_lengths,
            dtype=torch.long,
            device=self.device,
        )
        return speech_tokens.detach(), speech_token_lens.detach()

    def log_mel_spectrogram(self, audio: torch.Tensor):
        return self.feature_extractor(audio)
