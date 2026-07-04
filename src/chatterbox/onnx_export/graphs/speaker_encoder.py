from __future__ import annotations

import torch

from ...models.speaker.exportable import SpeakerEncoderExport
from ..constants import GRAPH_SPEAKER_ENCODER
from ..dynamic_shapes import SPEAKER_ENCODER_DYNAMIC_SHAPES
from ..graph_spec import GraphSpec
from ..names import SPEAKER_ENCODER

input_names = ["fbank"]
output_names = ["embedding"]
dynamic_shapes = SPEAKER_ENCODER_DYNAMIC_SHAPES


def make_module(model):
    return SpeakerEncoderExport(model.speaker_encoder)


def make_dummy_inputs(batch: int = 1, frames: int = 256):
    return (torch.randn(batch, frames, 80, dtype=torch.float32),)


SPEAKER_ENCODER_SPEC = GraphSpec(
    name=GRAPH_SPEAKER_ENCODER,
    filename=SPEAKER_ENCODER,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    make_module=make_module,
    make_dummy_inputs=make_dummy_inputs,
    input_dtypes={"fbank": "float32"},
    output_dtypes={"embedding": "float32"},
)
