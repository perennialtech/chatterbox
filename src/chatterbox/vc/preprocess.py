from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..audio import S3_SR, S3GEN_SR, load_audio_mono
from ..models.s3tokenizer.features import S3TokenizerLogMel
from ..models.speaker.features import extract_fbank_features


def load_wav_16k(
    path: str | Path, device: str = "cpu", max_len: int | None = None
) -> torch.Tensor:
    return load_audio_mono(path, S3_SR, device, max_len=max_len).unsqueeze(0)


def load_wav_24k(
    path: str | Path, device: str = "cpu", max_len: int | None = None
) -> torch.Tensor:
    return load_audio_mono(path, S3GEN_SR, device, max_len=max_len).unsqueeze(0)


@torch.inference_mode()
def compute_s3_log_mel(audio_16k: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    extractor = S3TokenizerLogMel().eval()
    audio_16k = audio_16k.detach().cpu().float()
    log_mel = extractor(audio_16k)
    mel_lengths = torch.full(
        (log_mel.size(0),),
        log_mel.size(-1),
        dtype=torch.long,
    )
    return (
        np.ascontiguousarray(log_mel.cpu().numpy().astype(np.float32)),
        np.ascontiguousarray(mel_lengths.cpu().numpy().astype(np.int32)),
    )


@torch.inference_mode()
def compute_fbank(audio_16k: torch.Tensor) -> np.ndarray:
    audio_16k = audio_16k.detach().cpu().float()
    fbank = extract_fbank_features(audio_16k)
    return np.ascontiguousarray(fbank.cpu().numpy().astype(np.float32))
