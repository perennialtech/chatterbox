import torch

from ..dynamic_axes import SPEAKER_ENCODER_DYNAMIC_AXES

input_names = ["fbank", "fbank_lengths"]
output_names = ["embedding"]
dynamic_axes = SPEAKER_ENCODER_DYNAMIC_AXES


class SpeakerEncoderExport(torch.nn.Module):
    def __init__(self, speaker_encoder: torch.nn.Module):
        super().__init__()
        self.speaker_encoder = speaker_encoder

    def forward(self, fbank, fbank_lengths):
        return self.speaker_encoder(fbank)


def make_dummy_inputs(batch: int = 1, frames: int = 256):
    return (
        torch.randn(batch, frames, 80, dtype=torch.float32),
        torch.full((batch,), frames, dtype=torch.long),
    )
