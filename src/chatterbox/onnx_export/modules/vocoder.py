import torch

from ..dynamic_shapes import VOCODER_DYNAMIC_SHAPES

input_names = ["speech_feat", "source_phase", "source_noise"]
output_names = ["wav", "source"]
dynamic_axes = VOCODER_DYNAMIC_SHAPES


class VocoderExport(torch.nn.Module):
    def __init__(self, vocoder: torch.nn.Module):
        super().__init__()
        self.vocoder = vocoder

    def forward(self, speech_feat, source_phase, source_noise):
        return self.vocoder(
            speech_feat=speech_feat,
            source_phase=source_phase,
            source_noise=source_noise,
        )


def make_dummy_inputs(
    batch: int = 1,
    mel_frames: int = 64,
    source_hop: int = 120,
    harmonics: int = 9,
    dtype=torch.float32,
):
    return (
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.zeros(batch, harmonics, 1, dtype=dtype),
        torch.randn(batch, harmonics, mel_frames * source_hop, dtype=dtype),
    )
