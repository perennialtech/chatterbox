from __future__ import annotations

import torch

from ..constants import GRAPH_FLOW_DECODER_MEANFLOW2, MEANFLOW_T_SPAN
from ..dynamic_shapes import FLOW_DECODER_DYNAMIC_SHAPES
from ..graph_spec import GraphSpec
from ..names import FLOW_DECODER_MEANFLOW2

input_names = ["noise", "mask", "mu", "spks", "cond"]
output_names = ["mel"]
dynamic_shapes = FLOW_DECODER_DYNAMIC_SHAPES


class FlowDecoderMeanflow2Export(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder
        self.register_buffer(
            "t_span", torch.tensor(MEANFLOW_T_SPAN, dtype=torch.float32)
        )

    def forward(self, noise, mask, mu, spks, cond):
        x = noise
        for i in range(2):
            t = self.t_span[i].to(dtype=x.dtype).expand(x.size(0))
            r = self.t_span[i + 1].to(dtype=x.dtype).expand(x.size(0))
            dxdt = self.decoder.estimator.forward_export(
                x=x,
                mask=mask,
                mu=mu,
                spks=spks,
                cond=cond,
                t=t,
                r=r,
            )
            x = x + (r[:1] - t[:1]) * dxdt
        return x


def make_module(model):
    return FlowDecoderMeanflow2Export(model.flow.decoder)


def make_dummy_inputs(batch: int = 1, mel_frames: int = 64, dtype=torch.float32):
    return (
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.ones(batch, 1, mel_frames, dtype=dtype),
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.randn(batch, 80, dtype=dtype),
        torch.randn(batch, 80, mel_frames, dtype=dtype),
    )


FLOW_DECODER_MEANFLOW2_SPEC = GraphSpec(
    name=GRAPH_FLOW_DECODER_MEANFLOW2,
    filename=FLOW_DECODER_MEANFLOW2,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    make_module=make_module,
    make_dummy_inputs=make_dummy_inputs,
    input_dtypes={
        "noise": "float32",
        "mask": "float32",
        "mu": "float32",
        "spks": "float32",
        "cond": "float32",
    },
    output_dtypes={"mel": "float32"},
)
