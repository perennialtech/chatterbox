from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from .errors import VoiceConditioningError


def _as_numpy(value, name: str, dtype: np.dtype) -> np.ndarray:
    if value is None:
        raise VoiceConditioningError(f"{name} is required")
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.dtype != dtype:
        array = array.astype(dtype)
    return np.ascontiguousarray(array)


@dataclass(frozen=True)
class VoiceConditionTensors:
    prompt_token: np.ndarray
    prompt_token_len: np.ndarray
    prompt_feat: np.ndarray
    embedding: np.ndarray

    @classmethod
    def from_mapping(
        cls, data: Mapping[str, object] | "VoiceConditionTensors"
    ) -> "VoiceConditionTensors":
        if isinstance(data, VoiceConditionTensors):
            data.validate()
            return data

        missing = {
            "prompt_token",
            "prompt_token_len",
            "prompt_feat",
            "embedding",
        } - set(data)
        if missing:
            raise VoiceConditioningError(
                f"Missing voice conditioning tensors: {sorted(missing)}"
            )

        condition = cls(
            prompt_token=_as_numpy(data["prompt_token"], "prompt_token", np.int64),
            prompt_token_len=_as_numpy(
                data["prompt_token_len"], "prompt_token_len", np.int64
            ),
            prompt_feat=_as_numpy(data["prompt_feat"], "prompt_feat", np.float32),
            embedding=_as_numpy(data["embedding"], "embedding", np.float32),
        )
        condition.validate()
        return condition

    def validate(self) -> None:
        if self.prompt_token.dtype != np.int64:
            raise VoiceConditioningError("prompt_token must be int64")
        if self.prompt_token_len.dtype != np.int64:
            raise VoiceConditioningError("prompt_token_len must be int64")
        if self.prompt_feat.dtype != np.float32:
            raise VoiceConditioningError("prompt_feat must be float32")
        if self.embedding.dtype != np.float32:
            raise VoiceConditioningError("embedding must be float32")

        if self.prompt_token.ndim != 2 or self.prompt_token.shape[0] != 1:
            raise VoiceConditioningError("prompt_token must have shape [1, P]")
        if self.prompt_token_len.shape != (1,):
            raise VoiceConditioningError("prompt_token_len must have shape [1]")
        if (
            self.prompt_feat.ndim != 3
            or self.prompt_feat.shape[0] != 1
            or self.prompt_feat.shape[2] != 80
        ):
            raise VoiceConditioningError("prompt_feat must have shape [1, M, 80]")
        if self.embedding.shape != (1, 192):
            raise VoiceConditioningError("embedding must have shape [1, 192]")

        prompt_tokens = int(self.prompt_token_len[0])
        if prompt_tokens <= 0 or prompt_tokens > self.prompt_token.shape[1]:
            raise VoiceConditioningError(
                "prompt_token_len is outside prompt_token bounds"
            )

        required_prompt_mels = 2 * self.prompt_token.shape[1]
        if self.prompt_feat.shape[1] != required_prompt_mels:
            raise VoiceConditioningError(
                f"prompt_feat has {self.prompt_feat.shape[1]} frames but {required_prompt_mels} are required"
            )

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "prompt_token": self.prompt_token,
            "prompt_token_len": self.prompt_token_len,
            "prompt_feat": self.prompt_feat,
            "embedding": self.embedding,
        }

    def to_torch(self, device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
        return {
            "prompt_token": torch.from_numpy(self.prompt_token).to(
                device=device, dtype=torch.long
            ),
            "prompt_token_len": torch.from_numpy(self.prompt_token_len).to(
                device=device, dtype=torch.long
            ),
            "prompt_feat": torch.from_numpy(self.prompt_feat).to(
                device=device, dtype=dtype
            ),
            "embedding": torch.from_numpy(self.embedding).to(
                device=device, dtype=dtype
            ),
        }
