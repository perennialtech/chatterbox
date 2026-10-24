"""Fast tokenizer paths retain the original integer codes and attention output."""

import pytest
import torch

from chatterbox.models.s3tokenizer.architecture import (
    FSMNMultiHeadAttention,
    FSQCodebook,
    ModelConfig,
)
from chatterbox.models.s3tokenizer.model import S3Tokenizer


@pytest.mark.parametrize("frames", [1, 25, 250])
def test_vectorized_fsq_matches_integer_reference(frames):
    torch.manual_seed(29)
    codebook = FSQCodebook(dim=32)
    hidden = torch.randn(1, frames, 32)
    result = codebook.encode(hidden)
    projected = codebook.project_down(hidden).float().tanh() * 0.9990000128746033
    codes = (projected.round() + 1).to(torch.int32).reshape(-1, 8)
    expected = codes[:, 0] * codebook.powers[0]
    for i in range(1, 8):
        expected += codes[:, i] * codebook.powers[i]
    assert torch.equal(result, expected.reshape(1, frames))


@pytest.mark.parametrize("frames", [1, 25, 251])
def test_sdpa_matches_explicit_attention_with_padding(frames):
    torch.manual_seed(41)
    attn = FSMNMultiHeadAttention(32, 4, use_sdpa=False).eval()
    x = torch.randn(1, frames, 32)
    bias = torch.zeros(1, 1, 1, frames)
    bias[..., frames // 2 :] = -1e10
    with torch.inference_mode():
        expected = attn(x, mask=bias)
        attn.use_sdpa = True
        actual = attn(x, mask=bias)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_default_tokenizer_uses_sdpa():
    assert ModelConfig().use_sdpa is True


@pytest.mark.parametrize("mel_frames", [1, 4, 5, 17, 300])
def test_unpadded_tokenizer_length_uses_output_width(mel_frames, monkeypatch):
    tokenizer = S3Tokenizer(
        "speech_tokenizer_v2_25hz",
        ModelConfig(n_audio_state=32, n_audio_head=4, n_audio_layer=1),
    )
    expected_length = (mel_frames + 3) // 4

    def fake_quantize(mel, mel_len):
        return torch.zeros(1, expected_length, dtype=torch.int32), torch.tensor(
            [expected_length]
        )

    monkeypatch.setattr(tokenizer, "quantize", fake_quantize)
    # No tensor-to-CPU copy is required to know the length of an unpadded chunk.
    monkeypatch.setattr(
        torch.Tensor,
        "cpu",
        lambda self: (_ for _ in ()).throw(AssertionError("GPU sync")),
    )
    tokens, length = tokenizer._quantize_mel_full(torch.zeros(1, 128, mel_frames))
    assert length == expected_length
    assert tokens.shape == (expected_length,)
