import torch

from ..audio import SPEECH_VOCAB_SIZE

SOS = SPEECH_VOCAB_SIZE
EOS = SPEECH_VOCAB_SIZE + 1


def drop_invalid_tokens(x) -> torch.Tensor:
    """Return valid speech-token IDs from a single token sequence."""
    x = torch.as_tensor(x)

    if x.ndim == 1:
        tokens = x
    elif x.ndim == 2 and x.size(0) == 1:
        tokens = x.squeeze(0)
    else:
        raise ValueError("only batch size of one is supported")

    return tokens[(tokens >= 0) & (tokens < SPEECH_VOCAB_SIZE)]
