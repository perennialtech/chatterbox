MEANFLOW_T_SPAN = (0.0, 0.5, 1.0)

GRAPH_S3_TOKENIZER_LOG_MEL = "s3_tokenizer_log_mel"
GRAPH_S3_TOKENIZER_QUANTIZER = "s3_tokenizer_quantizer"
GRAPH_SPEAKER_ENCODER = "speaker_encoder"
GRAPH_REFERENCE_MEL_24K = "reference_mel_24k"
GRAPH_TOKEN_TO_MU = "token_to_mu"
GRAPH_FLOW_DECODER_MEANFLOW2 = "flow_decoder_meanflow2"
GRAPH_VOCODER_HIFT = "vocoder_hift"


def token_to_mu_graph_name(token_bucket: int) -> str:
    return f"{GRAPH_TOKEN_TO_MU}_{int(token_bucket)}tok"


def flow_decoder_graph_name(mel_bucket: int) -> str:
    return f"{GRAPH_FLOW_DECODER_MEANFLOW2}_{int(mel_bucket)}mel"


def vocoder_graph_name(mel_bucket: int) -> str:
    return f"{GRAPH_VOCODER_HIFT}_{int(mel_bucket)}mel"
