import numpy as np


def bucket_length(length: int, buckets: tuple[int, ...]) -> int:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    return ((length + 511) // 512) * 512


def pad_tokens(tokens: np.ndarray, bucket: int, value: int = 0) -> np.ndarray:
    if tokens.shape[-1] >= bucket:
        return tokens[..., :bucket]
    pad_width = [(0, 0)] * tokens.ndim
    pad_width[-1] = (0, bucket - tokens.shape[-1])
    return np.pad(tokens, pad_width, mode="constant", constant_values=value)


def pad_mels(mels: np.ndarray, bucket: int) -> np.ndarray:
    if mels.shape[-1] >= bucket:
        return mels[..., :bucket]
    pad_width = [(0, 0)] * mels.ndim
    pad_width[-1] = (0, bucket - mels.shape[-1])
    return np.pad(mels, pad_width, mode="constant")
