import torch

from chatterbox.models.s3tokenizer import S3Tokenizer


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
