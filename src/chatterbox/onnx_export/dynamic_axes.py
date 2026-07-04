S3_TOKENIZER_DYNAMIC_AXES = {
    "log_mel": {0: "batch", 2: "mel_frames_16k"},
    "mel_lengths": {0: "batch"},
    "speech_tokens": {0: "batch", 1: "tokens"},
    "speech_token_lengths": {0: "batch"},
}

SPEAKER_ENCODER_DYNAMIC_AXES = {
    "fbank": {0: "batch", 1: "fbank_frames"},
    "fbank_lengths": {0: "batch"},
    "embedding": {0: "batch"},
}

REFERENCE_MEL_DYNAMIC_AXES = {
    "wav_24k": {0: "batch", 1: "samples"},
    "prompt_feat": {0: "batch", 1: "mel_frames"},
    "prompt_feat_len": {0: "batch"},
}

TOKEN_TO_MU_DYNAMIC_AXES = {
    "prompt_token": {0: "batch", 1: "prompt_tokens"},
    "prompt_token_len": {0: "batch"},
    "speech_token": {0: "batch", 1: "speech_tokens"},
    "speech_token_len": {0: "batch"},
    "mu": {0: "batch", 2: "total_mel_frames"},
    "mask": {0: "batch", 2: "total_mel_frames"},
    "prompt_mel_len": {0: "batch"},
    "output_mel_len": {0: "batch"},
}

CONDITIONAL_DECODER_DYNAMIC_AXES = {
    "x": {0: "batch", 2: "mel_frames"},
    "mask": {0: "batch", 2: "mel_frames"},
    "mu": {0: "batch", 2: "mel_frames"},
    "spks": {0: "batch"},
    "cond": {0: "batch", 2: "mel_frames"},
    "t": {0: "batch"},
    "r": {0: "batch"},
    "dxdt": {0: "batch", 2: "mel_frames"},
}

FLOW_DECODER_DYNAMIC_AXES = {
    "noise": {0: "batch", 2: "mel_frames"},
    "mask": {0: "batch", 2: "mel_frames"},
    "mu": {0: "batch", 2: "mel_frames"},
    "spks": {0: "batch"},
    "cond": {0: "batch", 2: "mel_frames"},
    "mel": {0: "batch", 2: "mel_frames"},
}

VOCODER_DYNAMIC_AXES = {
    "speech_feat": {0: "batch", 2: "mel_frames"},
    "source_phase": {0: "batch"},
    "source_noise": {0: "batch", 2: "source_samples"},
    "wav": {0: "batch", 1: "samples"},
    "source": {0: "batch", 2: "samples"},
}
