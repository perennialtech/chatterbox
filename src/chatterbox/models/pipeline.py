from __future__ import annotations

import torch

from chatterbox.audio import AudioProcessor
from chatterbox.device import Runtime
from chatterbox.models.checkpoint import PipelineConfig
from chatterbox.models.conditioning.reference import ReferenceEncoder
from chatterbox.models.melgen import TokenToMelModel
from chatterbox.models.tokenizer import S3_SR, SourceTokenizer
from chatterbox.models.vocoder import AudioPostprocessor, HiFTGenerator
from chatterbox.types import AudioInput, ReferenceConditioning, TokenBatch


class VoiceConversionPipeline(torch.nn.Module):
    def __init__(self, *, config: PipelineConfig, runtime: Runtime):
        super().__init__()
        self.config = config
        self.runtime = runtime
        self.audio = AudioProcessor(runtime)

        self.tokenizer = SourceTokenizer()
        self.reference_encoder = ReferenceEncoder(
            tokenizer=self.tokenizer,
            runtime=runtime,
            config=config,
        )
        self.mel_generator = TokenToMelModel()
        self.vocoder = HiFTGenerator(sampling_rate=config.sample_rate)
        self.postprocess = AudioPostprocessor(sample_rate=config.sample_rate)

    @torch.inference_mode()
    def encode_reference(self, target: AudioInput) -> ReferenceConditioning:
        return self.reference_encoder.encode(target)

    @torch.inference_mode()
    def convert(
        self,
        source: AudioInput,
        ref: ReferenceConditioning,
        *,
        steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        wav_16k = self.audio.load(source, sample_rate=S3_SR)
        tokens = self.tokenizer.encode(wav_16k)
        tokens = TokenBatch(
            tokens=tokens.tokens.to(device=self.runtime.device, dtype=torch.long),
            lengths=tokens.lengths.to(device=self.runtime.device, dtype=torch.long),
        )

        mels = self.mel_generator.generate(
            tokens,
            ref,
            steps=steps,
            generator=generator,
        )
        wav, _ = self.vocoder.inference(mels.mels.to(dtype=self.runtime.compute_dtype))
        return self.postprocess(wav)

    def compile_for_inference(self) -> "VoiceConversionPipeline":
        self.mel_generator.encoder = torch.compile(
            self.mel_generator.encoder,
            mode="default",
            dynamic=True,
        )
        self.mel_generator.sampler.estimator = torch.compile(
            self.mel_generator.sampler.estimator,
            mode="default",
            dynamic=True,
        )
        self.vocoder.compile_for_inference()
        return self
