import torch

# Setting min=1 avoids edge cases during dynamo compilation
# where 0-length dimensions can crash the tensor tracer.
batch = torch.export.Dim("batch", min=1)
mel_frames = torch.export.Dim("mel_frames", min=1)
mel_frames_16k = torch.export.Dim("mel_frames_16k", min=1)
tokens = torch.export.Dim("tokens", min=1)
fbank_frames = torch.export.Dim("fbank_frames", min=1)
samples = torch.export.Dim("samples", min=1)
prompt_tokens = torch.export.Dim("prompt_tokens", min=1)
speech_tokens = torch.export.Dim("speech_tokens", min=1)
total_mel_frames = torch.export.Dim("total_mel_frames", min=1)
source_samples = torch.export.Dim("source_samples", min=1)

S3_TOKENIZER_DYNAMIC_SHAPES = {
    "log_mel": {0: batch, 2: mel_frames_16k},
    "mel_lengths": {0: batch},
    "speech_tokens": {0: batch, 1: tokens},
    "speech_token_lengths": {0: batch},
}

SPEAKER_ENCODER_DYNAMIC_SHAPES = {
    "fbank": {0: batch, 1: fbank_frames},
    "fbank_lengths": {0: batch},
    "embedding": {0: batch},
}

REFERENCE_MEL_DYNAMIC_SHAPES = {
    "wav_24k": {0: batch, 1: samples},
    "prompt_feat": {0: batch, 1: mel_frames},
    "prompt_feat_len": {0: batch},
}

TOKEN_TO_MU_DYNAMIC_SHAPES = {
    "prompt_token": {0: batch, 1: prompt_tokens},
    "prompt_token_len": {0: batch},
    "speech_token": {0: batch, 1: speech_tokens},
    "speech_token_len": {0: batch},
    "embedding": {0: batch},
    "mu": {0: batch, 2: total_mel_frames},
    "mask": {0: batch, 2: total_mel_frames},
    "spks": {0: batch},
    "prompt_mel_len": {0: batch},
    "output_mel_len": {0: batch},
}

CONDITIONAL_DECODER_DYNAMIC_SHAPES = {
    "x": {0: batch, 2: mel_frames},
    "mask": {0: batch, 2: mel_frames},
    "mu": {0: batch, 2: mel_frames},
    "spks": {0: batch},
    "cond": {0: batch, 2: mel_frames},
    "t": {0: batch},
    "r": {0: batch},
    "dxdt": {0: batch, 2: mel_frames},
}

FLOW_DECODER_DYNAMIC_SHAPES = {
    "noise": {0: batch, 2: mel_frames},
    "mask": {0: batch, 2: mel_frames},
    "mu": {0: batch, 2: mel_frames},
    "spks": {0: batch},
    "cond": {0: batch, 2: mel_frames},
    "mel": {0: batch, 2: mel_frames},
}

VOCODER_DYNAMIC_SHAPES = {
    "speech_feat": {0: batch, 2: mel_frames},
    "source_phase": {0: batch},
    "source_noise": {0: batch, 2: source_samples},
    "wav": {0: batch, 1: samples},
    "source": {0: batch, 2: samples},
}
