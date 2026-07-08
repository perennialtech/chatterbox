import pytest
import torch

from chatterbox.audio import S3_SR, S3_TOKEN_RATE
from chatterbox.models.s3tokenizer import S3Tokenizer
from chatterbox.models.s3tokenizer.architecture import ModelConfig


def test_s3tokenizer_prepare_audio_accepts_single_waveform_tensor():
    tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    prepared = tokenizer._prepare_audio(torch.zeros(160))

    assert len(prepared) == 1
    assert prepared[0].shape == (1, 160)


def test_s3tokenizer_prepare_audio_accepts_batched_waveform_tensor():
    tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    prepared = tokenizer._prepare_audio(torch.zeros(2, 160))

    assert len(prepared) == 2
    assert prepared[0].shape == (1, 160)
    assert prepared[1].shape == (1, 160)


@pytest.mark.parametrize("duration_seconds", [5, 15])
def test_s3tokenizer_token_lengths_follow_25hz_rate(duration_seconds):
    config = ModelConfig(
        n_audio_ctx=2048,
        n_audio_state=40,
        n_audio_head=4,
        n_audio_layer=1,
    )
    tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz", config=config).eval()

    wav = torch.zeros(duration_seconds * S3_SR)
    tokens, token_lens = tokenizer(wav)

    expected_tokens = duration_seconds * int(S3_TOKEN_RATE)
    assert abs(token_lens.item() - expected_tokens) <= 2
    assert tokens.shape == (1, token_lens.item())
