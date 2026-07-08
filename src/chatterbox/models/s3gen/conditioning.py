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

        condition = cls(
            prompt_token=tensor("prompt_token", torch.long),
            prompt_token_len=tensor("prompt_token_len", torch.long),
            prompt_feat=tensor("prompt_feat", dtype),
            embedding=tensor("embedding", dtype),
        )
        condition.validate()
        return condition

    def validate(self) -> None:
        if self.prompt_token.ndim != 2:
            raise ConditioningError("prompt_token must have shape [B, P]")
        if self.prompt_token_len.ndim != 1:
            raise ConditioningError("prompt_token_len must have shape [B]")
        if self.prompt_feat.ndim != 3:
            raise ConditioningError("prompt_feat must have shape [B, 2P, 80]")
        if self.embedding.ndim != 2:
            raise ConditioningError("embedding must have shape [B, 192]")

        batch_size = self.prompt_token.size(0)
        if self.prompt_token_len.size(0) != batch_size:
            raise ConditioningError("prompt_token_len batch size differs")
        if self.prompt_feat.size(0) != batch_size:
            raise ConditioningError("prompt_token and prompt_feat batch sizes differ")
        if self.embedding.size(0) != batch_size:
            raise ConditioningError("prompt_token and embedding batch sizes differ")

        if self.prompt_feat.size(-1) != 80:
            raise ConditioningError("prompt_feat must have 80 mel bins")
        if self.embedding.size(-1) != 192:
            raise ConditioningError("embedding must have 192 channels")

        if torch.any(self.prompt_token_len <= 0):
            raise ConditioningError("prompt_token_len values must be positive")
        if torch.any(self.prompt_token_len > self.prompt_token.size(1)):
            raise ConditioningError("prompt_token_len exceeds prompt_token width")

        expected_feat_width = self.prompt_token.size(1) * 2
        if self.prompt_feat.size(1) != expected_feat_width:
            raise ConditioningError(
                "prompt_feat width must be exactly 2x prompt_token width"
            )

    def trim_to_lengths(
        self,
        *,
        max_prompt_tokens: int | None = None,
    ) -> "S3ReferenceCondition":
        self.validate()

        if not torch.all(self.prompt_token_len == self.prompt_token_len[0]):
            raise ConditioningError(
                "batched references must use one shared prompt length"
            )

        prompt_token_len = int(self.prompt_token_len[0].detach().cpu())
        if max_prompt_tokens is not None:
            if max_prompt_tokens <= 0:
                raise ConditioningError("max_prompt_tokens must be positive")
            prompt_token_len = min(prompt_token_len, int(max_prompt_tokens))

        prompt_feat_len = prompt_token_len * 2
        prompt_token_len_tensor = torch.full_like(
            self.prompt_token_len,
            prompt_token_len,
        )

        trimmed = S3ReferenceCondition(
            prompt_token=self.prompt_token[:, :prompt_token_len].contiguous(),
            prompt_token_len=prompt_token_len_tensor,
            prompt_feat=self.prompt_feat[:, :prompt_feat_len].contiguous(),
            embedding=self.embedding,
        )
        trimmed.validate()
        return trimmed

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {
            "prompt_token": self.prompt_token,
            "prompt_token_len": self.prompt_token_len,
            "prompt_feat": self.prompt_feat,
            "embedding": self.embedding,
        }
