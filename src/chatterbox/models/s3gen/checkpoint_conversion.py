from collections.abc import Mapping

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


def convert_diffusers_transformer_keys(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    converted: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for old_suffix, local_suffix in _DIFFUSERS_TO_LOCAL_SUFFIXES.items():
            if key.endswith(old_suffix):
                new_key = key[: -len(old_suffix)] + local_suffix
                break
        converted[new_key] = value
    return converted
