from __future__ import annotations

import torch

from ...models.s3tokenizer.exportable import S3TokenizerQuantizerExport
from ..constants import GRAPH_S3_TOKENIZER_QUANTIZER
from ..dynamic_shapes import S3_TOKENIZER_DYNAMIC_SHAPES
from ..graph_spec import GraphSpec
from ..names import S3_TOKENIZER_QUANTIZER

input_names = ["log_mel", "mel_lengths"]
output_names = ["speech_tokens", "speech_token_lengths"]
dynamic_shapes = S3_TOKENIZER_DYNAMIC_SHAPES


def make_module(model):
    return S3TokenizerQuantizerExport(model.tokenizer)


def make_dummy_inputs(batch: int = 1, mel_frames: int = 256, n_mels: int = 128):
    return (
        torch.randn(batch, n_mels, mel_frames, dtype=torch.float32),
        torch.full((batch,), mel_frames, dtype=torch.int32),
    )


S3_TOKENIZER_QUANTIZER_SPEC = GraphSpec(
    name=GRAPH_S3_TOKENIZER_QUANTIZER,
    filename=S3_TOKENIZER_QUANTIZER,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    make_module=make_module,
    make_dummy_inputs=make_dummy_inputs,
    input_dtypes={"log_mel": "float32", "mel_lengths": "int32"},
    output_dtypes={"speech_tokens": "int32", "speech_token_lengths": "int32"},
)
