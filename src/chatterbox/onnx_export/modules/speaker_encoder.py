import torch

from ..dynamic_axes import SPEAKER_ENCODER_DYNAMIC_AXES

input_names = ["fbank", "fbank_lengths"]
output_names = ["embedding"]
dynamic_axes = SPEAKER_ENCODER_DYNAMIC_AXES


def make_dummy_inputs(batch: int = 1, frames: int = 256):
    return (
        torch.randn(batch, frames, 80, dtype=torch.float32),
        torch.full((batch,), frames, dtype=torch.long),
    )
