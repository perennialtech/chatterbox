from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch


class ConditioningError(ValueError):
    pass


@dataclass(frozen=True)
class S3ReferenceCondition:
    prompt_token: torch.Tensor
    prompt_token_len: torch.Tensor
    prompt_feat: torch.Tensor
    prompt_feat_len: torch.Tensor | None
    embedding: torch.Tensor

    @classmethod
    def from_mapping(
        cls,
        data: Mapping[str, object],
        device,
        dtype: torch.dtype,
    ) -> "S3ReferenceCondition":
        missing = {
            "prompt_token",
            "prompt_token_len",
            "prompt_feat",
            "embedding",
        } - set(data)
        if missing:
            raise ConditioningError(f"Missing reference tensors: {sorted(missing)}")

        def tensor(name: str, target_dtype: torch.dtype | None = None):
            value = data[name]
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            if not torch.is_tensor(value):
                raise ConditioningError(f"{name} must be a tensor")
            return (
                value.to(device=device, dtype=target_dtype)
                if target_dtype
                else value.to(device=device)
            )

        prompt_token = tensor("prompt_token", torch.long)
        prompt_token_len = tensor("prompt_token_len", torch.long)
        prompt_feat = tensor("prompt_feat", dtype)
        embedding = tensor("embedding", dtype)

        prompt_feat_len = None
        if data.get("prompt_feat_len") is not None:
            prompt_feat_len = tensor("prompt_feat_len", torch.long)

        condition = cls(
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
        )
        condition.validate()
        return condition

    def validate(self) -> None:
        if self.prompt_token.ndim != 2:
            raise ConditioningError("prompt_token must have shape [B, P]")
        if self.prompt_token_len.ndim != 1:
            raise ConditioningError("prompt_token_len must have shape [B]")
        if self.prompt_feat.ndim != 3:
            raise ConditioningError("prompt_feat must have shape [B, T, 80]")
        if self.embedding.ndim != 2:
            raise ConditioningError("embedding must have shape [B, 192]")
        if self.prompt_token.size(0) != self.prompt_feat.size(0):
            raise ConditioningError("prompt_token and prompt_feat batch sizes differ")
        if self.prompt_token.size(0) != self.embedding.size(0):
            raise ConditioningError("prompt_token and embedding batch sizes differ")

    def as_dict(self) -> dict[str, torch.Tensor | None]:
        return {
            "prompt_token": self.prompt_token,
            "prompt_token_len": self.prompt_token_len,
            "prompt_feat": self.prompt_feat,
            "prompt_feat_len": self.prompt_feat_len,
            "embedding": self.embedding,
        }


def build_decoder_condition(
    prompt_feat: torch.Tensor,
    total_mel_len: int,
    feat_dim: int = 80,
) -> torch.Tensor:
    batch = prompt_feat.size(0)
    cond = torch.zeros(
        batch,
        total_mel_len,
        feat_dim,
        device=prompt_feat.device,
        dtype=prompt_feat.dtype,
    )
    prompt_len = min(prompt_feat.size(1), total_mel_len)
    cond[:, :prompt_len] = prompt_feat[:, :prompt_len]
    return cond.transpose(1, 2).contiguous()
