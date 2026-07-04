import torch

from ..dynamic_shapes import SPEAKER_ENCODER_DYNAMIC_SHAPES

# input_names = ["fbank", "fbank_lengths"]
input_names = ["fbank"]
output_names = ["embedding"]
dynamic_axes = SPEAKER_ENCODER_DYNAMIC_SHAPES


def make_dummy_inputs(batch: int = 1, frames: int = 256):
    return (
        torch.randn(batch, frames, 80, dtype=torch.float32),
        torch.full((batch,), frames, dtype=torch.long),
    )
