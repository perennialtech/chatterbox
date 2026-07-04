from __future__ import annotations

import torch

from ..constants import GRAPH_VOCODER_HIFT
from ..dynamic_shapes import VOCODER_DYNAMIC_SHAPES
from ..graph_spec import GraphSpec
from ..names import VOCODER_HIFT

input_names = ["speech_feat", "source_phase", "source_noise"]
output_names = ["wav", "source"]
dynamic_shapes = VOCODER_DYNAMIC_SHAPES


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


def make_module(model):
    model.mel2wav.optimize_for_inference()
    return VocoderExport(model.mel2wav)


def make_dummy_inputs(
    batch: int = 1,
    mel_frames: int = 64,
    source_hop: int = 480,
    harmonics: int = 9,
    dtype=torch.float32,
):
    return (
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.zeros(batch, harmonics, 1, dtype=dtype),
        torch.randn(batch, harmonics, mel_frames * source_hop, dtype=dtype),
    )


def make_model_dummy_inputs(model):
    return make_dummy_inputs(source_hop=model.mel2wav.source_hop)


VOCODER_HIFT_SPEC = GraphSpec(
    name=GRAPH_VOCODER_HIFT,
    filename=VOCODER_HIFT,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    make_module=make_module,
    make_dummy_inputs=lambda: make_dummy_inputs(),
    input_dtypes={
        "speech_feat": "float32",
        "source_phase": "float32",
        "source_noise": "float32",
    },
    output_dtypes={"wav": "float32", "source": "float32"},
)
