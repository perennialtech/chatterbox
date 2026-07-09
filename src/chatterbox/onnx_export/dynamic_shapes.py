import torch

S3_TOKENIZER_MIN_SAMPLES = 201
S3_TOKENIZER_MIN_LOG_MEL_FRAMES = 1
SPEAKER_ENCODER_MIN_FRAMES = 1
REFERENCE_MEL_24K_MIN_SAMPLES = 721
REFERENCE_MEL_24K_MIN_FRAMES = 1


def _dim(name: str, minimum: int):
    return torch.export.Dim(name, min=minimum)


S3_TOKENIZER_LOG_MEL_DYNAMIC_SHAPES = {
    "wav_16k": {1: _dim("s3tok_wav_samples", S3_TOKENIZER_MIN_SAMPLES)},
    "log_mel": {2: _dim("s3tok_log_mel_frames", S3_TOKENIZER_MIN_LOG_MEL_FRAMES)},
}

S3_TOKENIZER_DYNAMIC_SHAPES = {
    "log_mel": {2: _dim("s3tok_log_mel_frames", S3_TOKENIZER_MIN_LOG_MEL_FRAMES)},
    "mel_lengths": {},
    "speech_tokens": {1: _dim("s3tok_speech_tokens_time", 1)},
}

SPEAKER_ENCODER_DYNAMIC_SHAPES = {
    "fbank": {1: _dim("speaker_fbank_frames", SPEAKER_ENCODER_MIN_FRAMES)},
}

REFERENCE_MEL_DYNAMIC_SHAPES = {
    "wav_24k": {1: _dim("refmel_wav_samples", REFERENCE_MEL_24K_MIN_SAMPLES)},
    "prompt_feat": {1: _dim("refmel_prompt_feat_frames", REFERENCE_MEL_24K_MIN_FRAMES)},
}

TOKEN_TO_MU_DYNAMIC_SHAPES = {}

FLOW_DECODER_DYNAMIC_SHAPES = {}

VOCODER_DYNAMIC_SHAPES = {}
