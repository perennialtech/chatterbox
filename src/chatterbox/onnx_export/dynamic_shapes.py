import torch


def _dim(name: str):
    return torch.export.Dim(name, min=1)


S3_TOKENIZER_DYNAMIC_SHAPES = {
    "log_mel": {0: _dim("s3tok_log_mel_batch"), 2: _dim("s3tok_log_mel_frames")},
    "mel_lengths": {0: _dim("s3tok_mel_lengths_batch")},
    "speech_tokens": {
        0: _dim("s3tok_speech_tokens_batch"),
        1: _dim("s3tok_speech_tokens_time"),
    },
    "speech_token_lengths": {0: _dim("s3tok_speech_token_lengths_batch")},
}

SPEAKER_ENCODER_DYNAMIC_SHAPES = {
    "fbank": {0: _dim("speaker_fbank_batch"), 1: _dim("speaker_fbank_frames")},
    "embedding": {0: _dim("speaker_embedding_batch")},
}

REFERENCE_MEL_DYNAMIC_SHAPES = {
    "wav_24k": {0: _dim("refmel_wav_batch"), 1: _dim("refmel_wav_samples")},
    "prompt_feat": {
        0: _dim("refmel_prompt_feat_batch"),
        1: _dim("refmel_prompt_feat_frames"),
    },
}

TOKEN_TO_MU_DYNAMIC_SHAPES = {
    "prompt_token": {
        0: _dim("tok2mu_prompt_token_batch"),
        1: _dim("tok2mu_prompt_tokens"),
    },
    "prompt_token_len": {0: _dim("tok2mu_prompt_token_len_batch")},
    "speech_token": {
        0: _dim("tok2mu_speech_token_batch"),
        1: _dim("tok2mu_speech_tokens"),
    },
    "speech_token_len": {0: _dim("tok2mu_speech_token_len_batch")},
    "embedding": {0: _dim("tok2mu_embedding_batch")},
    "mu": {0: _dim("tok2mu_mu_batch"), 2: _dim("tok2mu_total_mel_frames")},
    "mask": {0: _dim("tok2mu_mask_batch"), 2: _dim("tok2mu_mask_mel_frames")},
    "spks": {0: _dim("tok2mu_spks_batch")},
    "prompt_mel_len": {0: _dim("tok2mu_prompt_mel_len_batch")},
    "output_mel_len": {0: _dim("tok2mu_output_mel_len_batch")},
}

FLOW_DECODER_DYNAMIC_SHAPES = {
    "noise": {0: _dim("flow_noise_batch"), 2: _dim("flow_noise_mel_frames")},
    "mask": {0: _dim("flow_mask_batch"), 2: _dim("flow_mask_mel_frames")},
    "mu": {0: _dim("flow_mu_batch"), 2: _dim("flow_mu_mel_frames")},
    "spks": {0: _dim("flow_spks_batch")},
    "cond": {0: _dim("flow_cond_batch"), 2: _dim("flow_cond_mel_frames")},
    "mel": {0: _dim("flow_mel_batch"), 2: _dim("flow_mel_frames")},
}

VOCODER_DYNAMIC_SHAPES = {
    "speech_feat": {
        0: _dim("vocoder_speech_feat_batch"),
        2: _dim("vocoder_speech_feat_mel_frames"),
    },
    "source_phase": {0: _dim("vocoder_source_phase_batch")},
    "source_noise": {
        0: _dim("vocoder_source_noise_batch"),
        2: _dim("vocoder_source_noise_samples"),
    },
    "wav": {0: _dim("vocoder_wav_batch"), 1: _dim("vocoder_wav_samples")},
    "source": {
        0: _dim("vocoder_source_batch"),
        2: _dim("vocoder_source_samples"),
    },
}
