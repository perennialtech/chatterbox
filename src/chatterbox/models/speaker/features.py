from collections.abc import Iterable

import torch
import torchaudio.compliance.kaldi as Kaldi


def pad_list(xs: list[torch.Tensor], pad_value: float) -> torch.Tensor:
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs)
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i, x in enumerate(xs):
        pad[i, : x.size(0)] = x

    return pad


def extract_fbank_features(audio: torch.Tensor | Iterable[torch.Tensor]):
    if torch.is_tensor(audio):
        audio_iter = audio if audio.ndim == 2 else audio.unsqueeze(0)
    else:
        audio_iter = audio

    features = []
    feature_times = []
    feature_lengths = []
    for au in audio_iter:
        feature = Kaldi.fbank(au.unsqueeze(0), num_mel_bins=80)
        feature = feature - feature.mean(dim=0, keepdim=True)
        features.append(feature)
        feature_times.append(au.shape[0])
        feature_lengths.append(feature.shape[0])

    return pad_list(features, pad_value=0), feature_lengths, feature_times
