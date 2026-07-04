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
