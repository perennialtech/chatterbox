from __future__ import annotations

import torch

from ..buckets import VOCODER_MEL_BUCKETS
from ..constants import vocoder_graph_name
from ..dynamic_shapes import VOCODER_DYNAMIC_SHAPES
from ..graph_spec import ExportContext, GraphSpec
from ..names import vocoder_filename

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


def make_dummy_inputs(context: ExportContext, mel_frames: int):
    return (
        torch.randn(1, 80, mel_frames, dtype=context.dtype, device=context.device),
        torch.zeros(
            1,
            context.vocoder_harmonics,
            1,
            dtype=context.dtype,
            device=context.device,
        ),
        torch.randn(
            1,
            context.vocoder_harmonics,
            mel_frames * context.source_hop,
            dtype=context.dtype,
            device=context.device,
        ),
    )


def make_spec(mel_bucket: int) -> GraphSpec:
    return GraphSpec(
        name=vocoder_graph_name(mel_bucket),
        filename=vocoder_filename(mel_bucket),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        make_module=make_module,
        make_dummy_inputs=lambda context, mel_bucket=mel_bucket: make_dummy_inputs(
            context,
            mel_frames=mel_bucket,
        ),
        input_dtypes={
            "speech_feat": "float32",
            "source_phase": "float32",
            "source_noise": "float32",
        },
        output_dtypes={"wav": "float32", "source": "float32"},
    )


VOCODER_HIFT_BUCKET_SPECS = tuple(make_spec(bucket) for bucket in VOCODER_MEL_BUCKETS)
VOCODER_HIFT_SPEC = VOCODER_HIFT_BUCKET_SPECS[0]
