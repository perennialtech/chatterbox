from __future__ import annotations

import torch
import torch.nn.functional as F

from ...models.s3gen.utils.mask import make_pad_mask
from ..constants import GRAPH_TOKEN_TO_MU
from ..dynamic_shapes import TOKEN_TO_MU_DYNAMIC_SHAPES
from ..graph_spec import GraphSpec
from ..names import TOKEN_TO_MU

input_names = [
    "prompt_token",
    "prompt_token_len",
    "speech_token",
    "speech_token_len",
    "embedding",
]
output_names = ["mu", "mask", "spks", "prompt_mel_len", "output_mel_len"]
dynamic_shapes = TOKEN_TO_MU_DYNAMIC_SHAPES


class TokenToMuExport(torch.nn.Module):
    def __init__(self, flow: torch.nn.Module):
        super().__init__()
        self.input_embedding = flow.input_embedding
        self.encoder = flow.encoder
        self.encoder_proj = flow.encoder_proj
        self.spk_embed_affine_layer = flow.spk_embed_affine_layer
        self.vocab_size = flow.vocab_size

    def forward(
        self, prompt_token, prompt_token_len, speech_token, speech_token_len, embedding
    ):
        embedding = F.normalize(embedding, dim=1)
        spks = self.spk_embed_affine_layer(embedding)

        token = torch.cat([prompt_token, speech_token], dim=1).long()
        token = token.clamp(min=0, max=self.vocab_size - 1)
        token_len = prompt_token_len + speech_token_len
        token_mask = (~make_pad_mask(token_len, max_len=token.size(1))).unsqueeze(-1)
        embedded = self.input_embedding(token) * token_mask.to(
            dtype=self.input_embedding.weight.dtype
        )
        h, h_mask = self.encoder(embedded, token_len)
        mu = self.encoder_proj(h).transpose(1, 2).contiguous()
        mask = h_mask.to(dtype=mu.dtype)
        prompt_mel_len = prompt_token_len * 2
        output_mel_len = speech_token_len * 2
        return mu, mask, spks, prompt_mel_len, output_mel_len


def make_module(model):
    return TokenToMuExport(model.flow)


def make_dummy_inputs(
    batch: int = 1,
    prompt_tokens: int = 16,
    speech_tokens: int = 32,
    dtype=torch.float32,
):
    return (
        torch.zeros(batch, prompt_tokens, dtype=torch.long),
        torch.full((batch,), prompt_tokens, dtype=torch.long),
        torch.zeros(batch, speech_tokens, dtype=torch.long),
        torch.full((batch,), speech_tokens, dtype=torch.long),
        torch.randn(batch, 192, dtype=dtype),
    )


TOKEN_TO_MU_SPEC = GraphSpec(
    name=GRAPH_TOKEN_TO_MU,
    filename=TOKEN_TO_MU,
    input_names=input_names,
    output_names=output_names,
    dynamic_shapes=dynamic_shapes,
    make_module=make_module,
    make_dummy_inputs=make_dummy_inputs,
    input_dtypes={
        "prompt_token": "int64",
        "prompt_token_len": "int64",
        "speech_token": "int64",
        "speech_token_len": "int64",
        "embedding": "float32",
    },
    output_dtypes={
        "mu": "float32",
        "mask": "float32",
        "spks": "float32",
        "prompt_mel_len": "int64",
        "output_mel_len": "int64",
    },
)
