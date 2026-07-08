from collections.abc import Iterable

import torch
import torchaudio.compliance.kaldi as Kaldi

from ...audio.constants import S3_SR


def pad_list(xs: list[torch.Tensor], pad_value: float) -> torch.Tensor:
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i, x in enumerate(xs):
        pad[i, : x.size(0)] = x

    return pad


def extract_fbank_features(
    audio: torch.Tensor | Iterable[torch.Tensor],
) -> torch.Tensor:
    if torch.is_tensor(audio):
        audio_iter = audio if audio.ndim == 2 else audio.unsqueeze(0)
    else:
        audio_iter = audio

    features = []
    for au in audio_iter:
        feature = Kaldi.fbank(
            au.unsqueeze(0),
            num_mel_bins=80,
            sample_frequency=S3_SR,
        )
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature)

    return pad_list(features, pad_value=0)
