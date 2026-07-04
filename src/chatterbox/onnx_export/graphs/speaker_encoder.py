from __future__ import annotations

import torch

from ...models.speaker.exportable import SpeakerEncoderExport
from ..constants import GRAPH_SPEAKER_ENCODER
from ..dynamic_axes import SPEAKER_ENCODER_DYNAMIC_AXES
from ..graph_spec import GraphSpec
from ..names import SPEAKER_ENCODER

input_names = ["fbank", "fbank_lengths"]
output_names = ["embedding"]
dynamic_axes = SPEAKER_ENCODER_DYNAMIC_AXES


def make_module(model):
    return SpeakerEncoderExport(model.speaker_encoder)


def make_dummy_inputs(batch: int = 1, frames: int = 256):
    return (
        torch.randn(batch, frames, 80, dtype=torch.float32),
        torch.full((batch,), frames, dtype=torch.long),
    )


SPEAKER_ENCODER_SPEC = GraphSpec(
    name=GRAPH_SPEAKER_ENCODER,
    filename=SPEAKER_ENCODER,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes=dynamic_axes,
    make_module=make_module,
    make_dummy_inputs=make_dummy_inputs,
    input_dtypes={"fbank": "float32", "fbank_lengths": "int64"},
    output_dtypes={"embedding": "float32"},
)
