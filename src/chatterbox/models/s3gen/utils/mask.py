# Copyright (c) 2019 Shigeki Karita
#               2020 Mobvoi Inc (Binbin Zhang)
#               2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch


def build_attention_mask(
    masks: torch.Tensor,
    mode: str = "full",
    lookahead: int = 0,
) -> torch.Tensor:
    """Build a bool attention mask shaped [B, L, L].

    Args:
        masks: valid-position mask shaped [B, 1, L] or [B, L].
        mode: "full", "causal", or "limited_lookahead".
        lookahead: future positions allowed for "limited_lookahead".

    Returns:
        Bool mask where True means the key position is visible to the query.
    """
    if masks.dtype != torch.bool:
        raise TypeError("masks must be bool")

    if masks.ndim == 3:
        if masks.size(1) != 1:
            raise ValueError(f"3D masks must have shape [B, 1, L], got {masks.shape}")
        valid_keys = masks.squeeze(1)
    elif masks.ndim == 2:
        valid_keys = masks
    else:
        raise ValueError(
            f"masks must have shape [B, 1, L] or [B, L], got {masks.shape}"
        )

    if mode not in {"full", "causal", "limited_lookahead"}:
        raise ValueError(f"unsupported attention mask mode: {mode}")

    if lookahead < 0:
        raise ValueError(f"lookahead must be non-negative, got {lookahead}")

    batch_size, length = valid_keys.shape
    allowed = valid_keys[:, None, :].expand(batch_size, length, length)

    if mode == "full":
        return allowed

    idx = torch.arange(length, device=valid_keys.device)
    query_idx = idx[:, None]
    key_idx = idx[None, :]

    if mode == "causal":
        position_allowed = key_idx <= query_idx
    else:
        position_allowed = key_idx <= query_idx + lookahead

    return allowed & position_allowed.unsqueeze(0)


def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Make mask tensor containing indices of padded part.

    See description of make_non_pad_mask.

    Args:
        lengths (torch.Tensor): Batch of lengths (B,).
    Returns:
        torch.Tensor: Mask tensor containing indices of padded part.

    Examples:
        >>> lengths = [5, 3, 2]
        >>> make_pad_mask(lengths)
        masks = [[0, 0, 0, 0 ,0],
                 [0, 0, 0, 1, 1],
                 [0, 0, 1, 1, 1]]
    """
    lengths = lengths.long()
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask
