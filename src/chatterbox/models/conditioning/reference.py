from __future__ import annotations

import torch

from chatterbox.audio import AudioProcessor
from chatterbox.device import Runtime
from chatterbox.models.checkpoint import PipelineConfig
from chatterbox.models.conditioning.speaker_encoder import CAMPPlus
from chatterbox.models.melgen.mel import MelSpectrogram
from chatterbox.models.tokenizer import S3_SR, SourceTokenizer
from chatterbox.types import AudioInput, ReferenceConditioning

S3GEN_SR = 24_000


class ReferenceEncoder(torch.nn.Module):
    def __init__(
        self,
        *,
        tokenizer: SourceTokenizer,
        runtime: Runtime,
        config: PipelineConfig,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.runtime = runtime
        self.config = config
        self.audio = AudioProcessor(runtime)
        self.mel = MelSpectrogram(sample_rate=S3GEN_SR)
        self.speaker_encoder = CAMPPlus(memory_efficient=False)

    @torch.inference_mode()
    def encode(self, audio: AudioInput) -> ReferenceConditioning:
        max_24k = self.config.max_reference_seconds * S3GEN_SR
        wav_24k = self.audio.load(audio, sample_rate=S3GEN_SR, max_samples=max_24k)
        wav_16k = self.audio.load(
            audio,
            sample_rate=S3_SR,
            max_samples=self.config.max_reference_seconds * S3_SR,
        )

        prompt_mels = self.mel(wav_24k).to(dtype=self.runtime.compute_dtype)
        tokens = self.tokenizer.encode(wav_16k)
        speaker = self.speaker_encoder.inference(
            wav_16k.to(dtype=self.runtime.compute_dtype)
        )
        speaker = speaker.to(
            device=self.runtime.device, dtype=self.runtime.compute_dtype
        )

        token_len = tokens.lengths
        mel_len = token_len * 2
        max_mel_len = int(mel_len.max().item())

        prompt_tokens = tokens.tokens[:, : int(token_len.max().item())]
        prompt_mels = prompt_mels[:, :, :max_mel_len]

        return ReferenceConditioning(
            prompt_tokens=prompt_tokens.to(
                device=self.runtime.device, dtype=torch.long
            ),
            prompt_token_lengths=token_len.to(
                device=self.runtime.device, dtype=torch.long
            ),
            prompt_mels=prompt_mels,
            prompt_mel_lengths=mel_len.to(device=self.runtime.device, dtype=torch.long),
            speaker_embedding=speaker,
        )
