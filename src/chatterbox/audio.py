from functools import lru_cache

import torch
import torchaudio as ta


def _device_key(device) -> str:
    return str(torch.device(device))


@lru_cache(maxsize=100)
def get_resampler(src_sr: int, dst_sr: int, device_key: str):
    return ta.transforms.Resample(src_sr, dst_sr).to(torch.device(device_key))


def resample_audio(
    wav: torch.Tensor,
    src_sr: int,
    dst_sr: int,
    device,
) -> torch.Tensor:
    device_key = _device_key(device)
    wav = wav.to(device=torch.device(device_key), dtype=torch.float32)

    if src_sr == dst_sr:
        return wav

    return get_resampler(src_sr, dst_sr, device_key)(wav)


def load_audio_mono(
    path,
    sample_rate: int,
    device,
    max_len: int | None = None,
) -> torch.Tensor:
    wav, src_sr = ta.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    wav = resample_audio(wav, src_sr, sample_rate, device).squeeze(0)

    if max_len is not None:
        wav = wav[:max_len]

    return wav.contiguous()
