from __future__ import annotations

import torch

from ..constants import GRAPH_CONDITIONAL_DECODER_STEP
from ..dynamic_shapes import CONDITIONAL_DECODER_DYNAMIC_SHAPES
from ..graph_spec import GraphSpec
from ..names import CONDITIONAL_DECODER_STEP

input_names = ["x", "mask", "mu", "spks", "cond", "t", "r"]
output_names = ["dxdt"]
dynamic_shapes = CONDITIONAL_DECODER_DYNAMIC_SHAPES


class ConditionalDecoderStepExport(torch.nn.Module):
    def __init__(self, estimator: torch.nn.Module):
        super().__init__()
        self.estimator = estimator

    def forward(self, x, mask, mu, spks, cond, t, r):
        return self.estimator.forward_export(
            x=x,
            mask=mask,
            mu=mu,
            spks=spks,
            cond=cond,
            t=t,
            r=r,
        )


def make_module(model):
    return ConditionalDecoderStepExport(model.flow.decoder.estimator)


def make_dummy_inputs(batch: int = 1, mel_frames: int = 64, dtype=torch.float32):
    return (
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.ones(batch, 1, mel_frames, dtype=dtype),
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.randn(batch, 80, dtype=dtype),
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.zeros(batch, dtype=dtype),
        torch.full((batch,), 0.5, dtype=dtype),
    )


CONDITIONAL_DECODER_STEP_SPEC = GraphSpec(
    name=GRAPH_CONDITIONAL_DECODER_STEP,
    filename=CONDITIONAL_DECODER_STEP,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    make_module=make_module,
    make_dummy_inputs=make_dummy_inputs,
    input_dtypes={
        "x": "float32",
        "mask": "float32",
        "mu": "float32",
        "spks": "float32",
        "cond": "float32",
        "t": "float32",
        "r": "float32",
    },
    output_dtypes={"dxdt": "float32"},
)
