from __future__ import annotations

import torch
import torch.nn.functional as F

from chatterbox.models.melgen.config import MelGeneratorConfig
from chatterbox.models.melgen.flow import MeanFlowSampler
from chatterbox.models.melgen.masks import make_pad_mask
from chatterbox.models.s3gen.decoder import ConditionalDecoder
from chatterbox.models.s3gen.transformer.upsample_encoder import (
    UpsampleConformerEncoder,
)
from chatterbox.types import MelBatch, ReferenceConditioning, TokenBatch


class TokenToMelModel(torch.nn.Module):
    def __init__(self, config: MelGeneratorConfig = MelGeneratorConfig()):
        super().__init__()
        self.config = config
        self.input_embedding = torch.nn.Embedding(config.vocab_size, config.token_dim)
        self.speaker_projection = torch.nn.Linear(
            config.speaker_embedding_dim,
            config.projected_speaker_dim,
        )

        self.encoder = UpsampleConformerEncoder(
            output_size=512,
            attention_heads=8,
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            input_layer="linear",
            pos_enc_layer_type="rel_pos_espnet",
            selfattention_layer_type="rel_selfattn",
            input_size=512,
            use_cnn_module=False,
            macaron_style=False,
        )
        self.encoder_projection = torch.nn.Linear(512, config.mel_bins)

        estimator = ConditionalDecoder(
            in_channels=320,
            out_channels=80,
            causal=True,
            channels=[256],
            dropout=0.0,
            attention_head_dim=64,
            n_blocks=4,
            num_mid_blocks=12,
            num_heads=8,
            act_fn="gelu",
            meanflow=True,
        )
        self.sampler = MeanFlowSampler(estimator, steps=config.default_meanflow_steps)

    @torch.inference_mode()
    def generate(
        self,
        source: TokenBatch,
        ref: ReferenceConditioning,
        *,
        steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> MelBatch:
        tokens = torch.cat([ref.prompt_tokens, source.tokens], dim=1)
        token_lengths = ref.prompt_token_lengths + source.lengths

        valid = (~make_pad_mask(token_lengths, max_len=tokens.size(1))).unsqueeze(-1)
        embedded = self.input_embedding(tokens.long()) * valid.to(
            dtype=self.input_embedding.weight.dtype
        )

        encoded, masks = self.encoder(embedded, token_lengths)
        encoded_lengths = masks.sum(dim=-1).squeeze(1).long()

        prompt_mel_len = int(ref.prompt_mel_lengths.max().item())
        source_mel_len = int(source.lengths.max().item()) * 2
        target_mel_len = prompt_mel_len + source_mel_len

        encoded = encoded[:, :target_mel_len]
        encoded_lengths = torch.clamp(encoded_lengths, max=target_mel_len)

        mu = self.encoder_projection(encoded).transpose(1, 2).contiguous()

        cond = torch.zeros_like(mu)
        cond[:, :, :prompt_mel_len] = ref.prompt_mels[:, :, :prompt_mel_len]

        mask = (~make_pad_mask(encoded_lengths, max_len=mu.size(2))).unsqueeze(1).to(mu)
        speaker = F.normalize(ref.speaker_embedding, dim=1)
        speaker = self.speaker_projection(speaker)

        mels = self.sampler.sample(
            mu=mu,
            mask=mask,
            speaker=speaker,
            cond=cond,
            steps=steps,
            generator=generator,
        )
        mels = mels[:, :, prompt_mel_len : prompt_mel_len + source_mel_len].contiguous()

        return MelBatch(
            mels=mels,
            lengths=source.lengths.to(device=mels.device, dtype=torch.long) * 2,
        )
