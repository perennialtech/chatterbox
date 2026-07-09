S3_TOKENIZER_LOG_MEL = "s3_tokenizer_log_mel.onnx"
S3_TOKENIZER_QUANTIZER = "s3_tokenizer_quantizer.onnx"
SPEAKER_ENCODER = "speaker_encoder.onnx"
REFERENCE_MEL_24K = "reference_mel_24k.onnx"


def token_to_mu_filename(token_bucket: int) -> str:
    return f"token_to_mu_{int(token_bucket)}tok.onnx"


def flow_decoder_filename(mel_bucket: int) -> str:
    return f"flow_decoder_meanflow2_{int(mel_bucket)}mel.onnx"


def vocoder_filename(mel_bucket: int) -> str:
    return f"vocoder_hift_{int(mel_bucket)}mel.onnx"
