from collections.abc import Mapping
from dataclasses import dataclass

import torch

_DIFFUSERS_TO_LOCAL_SUFFIXES = {
    "attn1.to_q.weight": "attn1.to_q.weight",
    "attn1.to_q.bias": "attn1.to_q.bias",
    "attn1.to_k.weight": "attn1.to_k.weight",
    "attn1.to_k.bias": "attn1.to_k.bias",
    "attn1.to_v.weight": "attn1.to_v.weight",
    "attn1.to_v.bias": "attn1.to_v.bias",
    "attn1.to_out.0.weight": "attn1.to_out.0.weight",
    "attn1.to_out.0.bias": "attn1.to_out.0.bias",
    "norm1.weight": "norm1.weight",
    "norm1.bias": "norm1.bias",
    "norm3.weight": "norm3.weight",
    "norm3.bias": "norm3.bias",
    "ff.net.0.proj.weight": "ff.net.0.weight",
    "ff.net.0.proj.bias": "ff.net.0.bias",
    "ff.net.2.weight": "ff.net.3.weight",
    "ff.net.2.bias": "ff.net.3.bias",
}


class CheckpointLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class CheckpointValidationReport:
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    shape_mismatches: tuple[str, ...]


def convert_diffusers_transformer_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    source_for_key: dict[str, str] = {}

    for key, value in state_dict.items():
        new_key = key
        for old_suffix, local_suffix in _DIFFUSERS_TO_LOCAL_SUFFIXES.items():
            if key.endswith(old_suffix):
                new_key = key[: -len(old_suffix)] + local_suffix
                break

        if new_key in converted:
            raise ValueError(
                "checkpoint key conversion collision: "
                f"{source_for_key[new_key]!r} and {key!r} both map to {new_key!r}"
            )

        converted[new_key] = value
        source_for_key[new_key] = key

    converted.pop("tokenizer._mel_filters", None)
    converted.pop("tokenizer.window", None)

    return converted


def validate_checkpoint_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
) -> tuple[dict[str, torch.Tensor], CheckpointValidationReport]:
    converted = convert_diffusers_transformer_keys(state_dict)
    model_state = model.state_dict()

    missing_keys = tuple(key for key in model_state if key not in converted)
    unexpected_keys = tuple(key for key in converted if key not in model_state)

    shape_mismatches = []
    for key, value in converted.items():
        if key not in model_state:
            continue
        expected_shape = tuple(model_state[key].shape)
        actual_shape = tuple(value.shape)
        if expected_shape != actual_shape:
            shape_mismatches.append(
                f"{key}: expected {expected_shape}, got {actual_shape}"
            )

    report = CheckpointValidationReport(
        missing_keys=missing_keys,
        unexpected_keys=unexpected_keys,
        shape_mismatches=tuple(shape_mismatches),
    )

    if shape_mismatches or (strict and (missing_keys or unexpected_keys)):
        sections = []
        if missing_keys:
            sections.append("missing keys:\n  " + "\n  ".join(missing_keys))
        if unexpected_keys:
            sections.append("unexpected keys:\n  " + "\n  ".join(unexpected_keys))
        if shape_mismatches:
            sections.append("shape mismatches:\n  " + "\n  ".join(shape_mismatches))
        raise CheckpointLoadError("\n".join(sections))

    return converted, report


def load_converted_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, torch.Tensor],
    *,
    strict: bool = True,
):
    converted, report = validate_checkpoint_state_dict(
        model,
        state_dict,
        strict=strict,
    )
    load_result = model.load_state_dict(converted, strict=strict)
    return load_result, report
