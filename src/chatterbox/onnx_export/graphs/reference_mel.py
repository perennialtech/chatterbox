from __future__ import annotations

import torch

from ...audio import S3GEN_SR
from ...models.s3gen.mel import S3GenMelSpectrogram
from ..constants import GRAPH_REFERENCE_MEL_24K
from ..dynamic_shapes import REFERENCE_MEL_DYNAMIC_SHAPES
from ..graph_spec import ExportContext, GraphSpec
from ..names import REFERENCE_MEL_24K

input_names = ["wav_24k"]
output_names = ["prompt_feat"]
dynamic_shapes = REFERENCE_MEL_DYNAMIC_SHAPES


class ReferenceMel24kExport(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mel = S3GenMelSpectrogram(sampling_rate=S3GEN_SR)

    def forward(self, wav_24k):
        return self.mel(wav_24k).transpose(1, 2).contiguous()


def make_module(model):
    return ReferenceMel24kExport()


def make_dummy_inputs(context: ExportContext, samples: int = S3GEN_SR):
    return (
        torch.randn(
            1,
            samples,
            dtype=context.dtype,
            device=context.device,
        ),
    )


REFERENCE_MEL_24K_SPEC = GraphSpec(
    name=GRAPH_REFERENCE_MEL_24K,
    filename=REFERENCE_MEL_24K,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    make_module=make_module,
    make_dummy_inputs=make_dummy_inputs,
    input_dtypes={"wav_24k": "float32"},
    output_dtypes={"prompt_feat": "float32"},
)
