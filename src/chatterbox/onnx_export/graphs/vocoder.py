from __future__ import annotations

import torch

from ..buckets import VOCODER_MEL_BUCKETS
from ..constants import vocoder_graph_name
from ..dynamic_shapes import VOCODER_DYNAMIC_SHAPES
from ..graph_spec import GraphSpec
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


def make_dummy_inputs(
    mel_frames: int,
    source_hop: int = 480,
    harmonics: int = 9,
    dtype=torch.float32,
):
    return (
        torch.randn(1, 80, mel_frames, dtype=dtype),
        torch.zeros(1, harmonics, 1, dtype=dtype),
        torch.randn(1, harmonics, mel_frames * source_hop, dtype=dtype),
    )


def make_spec(mel_bucket: int) -> GraphSpec:
    return GraphSpec(
        name=vocoder_graph_name(mel_bucket),
        filename=vocoder_filename(mel_bucket),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        make_module=make_module,
        make_dummy_inputs=lambda mel_bucket=mel_bucket: make_dummy_inputs(
            mel_frames=mel_bucket
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
