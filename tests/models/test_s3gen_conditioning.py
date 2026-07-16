import pytest
import torch

from chatterbox.models.s3gen.conditioning import (ConditioningError,
                                                  S3ReferenceCondition)
from chatterbox.vc.conditioning import VoiceConditionTensors
from chatterbox.vc.errors import VoiceConditioningError


def _valid_condition(**overrides):
    data = {
        "prompt_token": torch.zeros(1, 3, dtype=torch.long),
        "prompt_token_len": torch.tensor([3], dtype=torch.long),
        "prompt_feat": torch.zeros(1, 6, 80),
        "embedding": torch.zeros(1, 192),
    }
    data.update(overrides)
    return S3ReferenceCondition(**data)


def test_reference_condition_accepts_valid_shapes():
    _valid_condition().validate()


@pytest.mark.parametrize(
    "field,value",
    [
        ("prompt_feat", torch.zeros(1, 6, 79)),
        ("embedding", torch.zeros(1, 191)),
        ("prompt_token_len", torch.zeros(1, 1, dtype=torch.long)),
        ("prompt_token_len", torch.tensor([0], dtype=torch.long)),
        ("prompt_token_len", torch.tensor([4], dtype=torch.long)),
        ("prompt_feat", torch.zeros(1, 5, 80)),
    ],
)
def test_reference_condition_rejects_invalid_shapes_and_lengths(field, value):
    condition = _valid_condition(**{field: value})
    with pytest.raises(ConditioningError):
        condition.validate()


def test_reference_condition_trims_to_prompt_token_length():
    condition = _valid_condition(
        prompt_token=torch.zeros(1, 4, dtype=torch.long),
        prompt_token_len=torch.tensor([3], dtype=torch.long),
        prompt_feat=torch.zeros(1, 8, 80),
    )

    trimmed = condition.trim_to_lengths()

    assert trimmed.prompt_token.shape == (1, 3)
    assert trimmed.prompt_feat.shape == (1, 6, 80)
    assert trimmed.prompt_token_len.tolist() == [3]


def test_reference_condition_caps_prompt_token_length():
    condition = _valid_condition(
        prompt_token=torch.zeros(1, 8, dtype=torch.long),
        prompt_token_len=torch.tensor([8], dtype=torch.long),
        prompt_feat=torch.zeros(1, 16, 80),
    )

    trimmed = condition.trim_to_lengths(max_prompt_tokens=5)

    assert trimmed.prompt_token.shape == (1, 5)
    assert trimmed.prompt_feat.shape == (1, 10, 80)
    assert trimmed.prompt_token_len.tolist() == [5]


def test_voice_condition_rejects_features_shorter_than_stored_prompt_width():
    with pytest.raises(VoiceConditioningError):
        VoiceConditionTensors.from_mapping(
            {
                "prompt_token": torch.zeros(1, 4, dtype=torch.long),
                "prompt_token_len": torch.tensor([3], dtype=torch.long),
                "prompt_feat": torch.zeros(1, 6, 80),
                "embedding": torch.zeros(1, 192),
            }
        )
