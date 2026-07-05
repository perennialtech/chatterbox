from __future__ import annotations

import json
from pathlib import Path

from ..onnx_export.constants import (GRAPH_CONDITIONAL_DECODER_STEP,
                                     GRAPH_FLOW_DECODER_MEANFLOW2,
                                     GRAPH_REFERENCE_MEL_24K,
                                     GRAPH_S3_TOKENIZER_QUANTIZER,
                                     GRAPH_SPEAKER_ENCODER, GRAPH_TOKEN_TO_MU,
                                     GRAPH_VOCODER_HIFT)
from .errors import TensorRTShapeError
from .types import ShapeRange

MAX_TOTAL_TOKENS = 3072
MAX_PROMPT_TOKENS = 250
MAX_SPEECH_TOKENS = MAX_TOTAL_TOKENS - MAX_PROMPT_TOKENS
MAX_MEL_FRAMES = 2 * MAX_TOTAL_TOKENS
MAX_REF_24K_SAMPLES = 240000
# MAX_SOURCE_LOGMEL_FRAMES = 8192
MAX_SOURCE_LOGMEL_FRAMES = 2048
MAX_FBANK_FRAMES = 2000
SOURCE_HOP = 480


DEFAULT_SHAPE_RANGES: dict[str, dict[str, ShapeRange]] = {
    GRAPH_S3_TOKENIZER_QUANTIZER: {
        "log_mel": ShapeRange(
            (1, 128, 4), (1, 128, 1024), (1, 128, MAX_SOURCE_LOGMEL_FRAMES)
        ),
        "mel_lengths": ShapeRange((1,), (1,), (1,)),
    },
    GRAPH_SPEAKER_ENCODER: {
        "fbank": ShapeRange((1, 16, 80), (1, 400, 80), (1, MAX_FBANK_FRAMES, 80)),
    },
    GRAPH_REFERENCE_MEL_24K: {
        "wav_24k": ShapeRange((1, 2400), (1, 144000), (1, MAX_REF_24K_SAMPLES)),
    },
    GRAPH_TOKEN_TO_MU: {
        "prompt_token": ShapeRange((1, 1), (1, 150), (1, MAX_PROMPT_TOKENS)),
        "prompt_token_len": ShapeRange((1,), (1,), (1,)),
        "speech_token": ShapeRange((1, 1), (1, 384), (1, MAX_SPEECH_TOKENS)),
        "speech_token_len": ShapeRange((1,), (1,), (1,)),
        "embedding": ShapeRange((1, 192), (1, 192), (1, 192)),
    },
    GRAPH_CONDITIONAL_DECODER_STEP: {
        "x": ShapeRange((1, 80, 2), (1, 80, 768), (1, 80, MAX_MEL_FRAMES)),
        "mask": ShapeRange((1, 1, 2), (1, 1, 768), (1, 1, MAX_MEL_FRAMES)),
        "mu": ShapeRange((1, 80, 2), (1, 80, 768), (1, 80, MAX_MEL_FRAMES)),
        "spks": ShapeRange((1, 80), (1, 80), (1, 80)),
        "cond": ShapeRange((1, 80, 2), (1, 80, 768), (1, 80, MAX_MEL_FRAMES)),
        "t": ShapeRange((1,), (1,), (1,)),
        "r": ShapeRange((1,), (1,), (1,)),
    },
    GRAPH_FLOW_DECODER_MEANFLOW2: {
        "noise": ShapeRange((1, 80, 2), (1, 80, 768), (1, 80, MAX_MEL_FRAMES)),
        "mask": ShapeRange((1, 1, 2), (1, 1, 768), (1, 1, MAX_MEL_FRAMES)),
        "mu": ShapeRange((1, 80, 2), (1, 80, 768), (1, 80, MAX_MEL_FRAMES)),
        "spks": ShapeRange((1, 80), (1, 80), (1, 80)),
        "cond": ShapeRange((1, 80, 2), (1, 80, 768), (1, 80, MAX_MEL_FRAMES)),
    },
    GRAPH_VOCODER_HIFT: {
        "speech_feat": ShapeRange((1, 80, 2), (1, 80, 768), (1, 80, MAX_MEL_FRAMES)),
        "source_phase": ShapeRange((1, 9, 1), (1, 9, 1), (1, 9, 1)),
        "source_noise": ShapeRange(
            (1, 9, 2 * SOURCE_HOP),
            (1, 9, 768 * SOURCE_HOP),
            (1, 9, MAX_MEL_FRAMES * SOURCE_HOP),
        ),
    },
}


def _shape_range_from_dict(data: dict) -> ShapeRange:
    return ShapeRange(tuple(data["min"]), tuple(data["opt"]), tuple(data["max"]))


def _validate_range(graph: str, name: str, value: ShapeRange) -> None:
    if not (len(value.min) == len(value.opt) == len(value.max)):
        raise TensorRTShapeError(f"{graph}.{name}: min/opt/max ranks differ")
    if value.min[0] != 1 or value.opt[0] != 1 or value.max[0] != 1:
        raise TensorRTShapeError(
            f"{graph}.{name}: TensorRT VC runtime supports batch size 1 only"
        )
    for mn, op, mx in zip(value.min, value.opt, value.max):
        if not (mn <= op <= mx):
            raise TensorRTShapeError(f"{graph}.{name}: invalid range {value}")


def _validate_token_budget(plan: dict[str, dict[str, ShapeRange]]) -> None:
    token_shapes = plan[GRAPH_TOKEN_TO_MU]
    prompt_max = token_shapes["prompt_token"].max[1]
    speech_max = token_shapes["speech_token"].max[1]
    total_max = prompt_max + speech_max
    if total_max > MAX_TOTAL_TOKENS:
        raise TensorRTShapeError(
            "token_to_mu prompt_token.max[1] + speech_token.max[1] must be "
            f"<= {MAX_TOTAL_TOKENS}; got {prompt_max} + {speech_max} = {total_max}"
        )


def load_shape_plan(path: Path | None = None) -> dict[str, dict[str, ShapeRange]]:
    plan = {graph: dict(inputs) for graph, inputs in DEFAULT_SHAPE_RANGES.items()}
    if path is not None:
        raw = json.loads(Path(path).read_text())
        for graph, inputs in raw.get("graphs", {}).items():
            if graph not in plan:
                raise TensorRTShapeError(f"Unknown graph in shape plan: {graph}")
            for name, value in inputs.items():
                plan[graph][name] = _shape_range_from_dict(value)

    for graph, inputs in plan.items():
        for name, value in inputs.items():
            _validate_range(graph, name, value)
    _validate_token_budget(plan)
    return plan


def shape_plan_to_jsonable(plan: dict[str, dict[str, ShapeRange]]) -> dict:
    return {
        graph: {name: shape.as_dict() for name, shape in inputs.items()}
        for graph, inputs in plan.items()
    }
