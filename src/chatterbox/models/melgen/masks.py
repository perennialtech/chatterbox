from __future__ import annotations

import torch


def lengths_to_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    idx = torch.arange(max_len, device=lengths.device)
    return idx.unsqueeze(0) < lengths.long().unsqueeze(1)


def make_pad_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    return ~lengths_to_mask(lengths, max_len)


def attention_bias(valid: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    return (~valid).to(dtype=dtype) * -1.0e4
