S3_TOKENIZER_QUANTIZER = "s3_tokenizer_quantizer.onnx"
SPEAKER_ENCODER = "speaker_encoder.onnx"
REFERENCE_MEL_24K = "reference_mel_24k.onnx"
TOKEN_TO_MU = "token_to_mu.onnx"
CONDITIONAL_DECODER_STEP = "conditional_decoder_step.onnx"
FLOW_DECODER_MEANFLOW2 = "flow_decoder_meanflow2.onnx"
VOCODER_HIFT = "vocoder_hift.onnx"


def bucketed_token_to_mu(bucket: int) -> str:
    return f"token_to_mu_tok{bucket:05d}.onnx"


def bucketed_decoder_step(mel_bucket: int) -> str:
    return f"conditional_decoder_step_mel{mel_bucket:05d}.onnx"


def bucketed_vocoder(mel_bucket: int) -> str:
    return f"vocoder_hift_mel{mel_bucket:05d}.onnx"
