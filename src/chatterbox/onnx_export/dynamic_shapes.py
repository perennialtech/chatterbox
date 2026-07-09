import torch


def _dim(name: str):
    return torch.export.Dim(name, min=1)


S3_TOKENIZER_LOG_MEL_DYNAMIC_SHAPES = {
    "wav_16k": {1: _dim("s3tok_wav_samples")},
    "log_mel": {2: _dim("s3tok_log_mel_frames")},
}

S3_TOKENIZER_DYNAMIC_SHAPES = {
    "log_mel": {2: _dim("s3tok_log_mel_frames")},
    "mel_lengths": {},
    "speech_tokens": {1: _dim("s3tok_speech_tokens_time")},
}

SPEAKER_ENCODER_DYNAMIC_SHAPES = {
    "fbank": {1: _dim("speaker_fbank_frames")},
}

REFERENCE_MEL_DYNAMIC_SHAPES = {
    "wav_24k": {1: _dim("refmel_wav_samples")},
    "prompt_feat": {1: _dim("refmel_prompt_feat_frames")},
}

TOKEN_TO_MU_DYNAMIC_SHAPES = {}

FLOW_DECODER_DYNAMIC_SHAPES = {}

VOCODER_DYNAMIC_SHAPES = {}
