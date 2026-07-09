from __future__ import annotations

import torch

from ..constants import GRAPH_S3_TOKENIZER_LOG_MEL
from ..dynamic_shapes import S3_TOKENIZER_LOG_MEL_DYNAMIC_SHAPES
from ..graph_spec import ExportContext, GraphSpec
from ..names import S3_TOKENIZER_LOG_MEL

input_names = ["wav_16k"]
output_names = ["log_mel"]
dynamic_shapes = S3_TOKENIZER_LOG_MEL_DYNAMIC_SHAPES


class S3TokenizerLogMelExport(torch.nn.Module):
    def __init__(self, feature_extractor: torch.nn.Module):
        super().__init__()
        self.feature_extractor = feature_extractor

    def forward(self, wav_16k):
        return self.feature_extractor(wav_16k)


def make_module(model):
    return S3TokenizerLogMelExport(model.tokenizer.feature_extractor)


def make_dummy_inputs(context: ExportContext, samples: int = 16000):
    return (
        torch.randn(
            1,
            samples,
            dtype=context.dtype,
            device=context.device,
        ),
    )


S3_TOKENIZER_LOG_MEL_SPEC = GraphSpec(
    name=GRAPH_S3_TOKENIZER_LOG_MEL,
    filename=S3_TOKENIZER_LOG_MEL,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    make_module=make_module,
    make_dummy_inputs=make_dummy_inputs,
    input_dtypes={"wav_16k": "float32"},
    output_dtypes={"log_mel": "float32"},
)
