from dataclasses import dataclass
from pathlib import Path


@dataclass
class OnnxSessions:
    token_to_mu: object | None = None
    conditional_decoder_step: object | None = None
    flow_decoder: object | None = None
    vocoder: object | None = None
    s3_tokenizer_quantizer: object | None = None
    speaker_encoder: object | None = None
    reference_mel_24k: object | None = None

    @classmethod
    def from_dir(
        cls, artifact_dir: Path, providers: list[str] | None = None
    ) -> "OnnxSessions":
        import onnxruntime as ort

        providers = providers or ["CPUExecutionProvider"]

        def load(name: str):
            path = artifact_dir / name
            return (
                ort.InferenceSession(str(path), providers=providers)
                if path.exists()
                else None
            )

        return cls(
            token_to_mu=load("token_to_mu.onnx"),
            conditional_decoder_step=load("conditional_decoder_step.onnx"),
            flow_decoder=load("flow_decoder_meanflow2.onnx"),
            vocoder=load("vocoder_hift.onnx"),
            s3_tokenizer_quantizer=load("s3_tokenizer_quantizer.onnx"),
            speaker_encoder=load("speaker_encoder.onnx"),
            reference_mel_24k=load("reference_mel_24k.onnx"),
        )
