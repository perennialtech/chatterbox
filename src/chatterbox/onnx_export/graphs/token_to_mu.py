from __future__ import annotations

import torch
import torch.nn.functional as F

from ...models.s3gen.utils.mask import make_pad_mask
from ..buckets import TOKEN_TO_MU_TOKEN_BUCKETS
from ..constants import token_to_mu_graph_name
from ..dynamic_shapes import TOKEN_TO_MU_DYNAMIC_SHAPES
from ..graph_spec import GraphSpec
from ..names import token_to_mu_filename

input_names = [
    "token",
    "prompt_token_len",
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

    def forward(self, token, prompt_token_len, speech_token_len, embedding):
        embedding = F.normalize(embedding, dim=1)
        spks = self.spk_embed_affine_layer(embedding)

        token = token.long()
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
    token_bucket: int,
    prompt_tokens: int = 16,
    dtype=torch.float32,
):
    speech_tokens = max(1, token_bucket - prompt_tokens)
    token = torch.zeros(1, token_bucket, dtype=torch.int32)
    token[0, prompt_tokens : prompt_tokens + speech_tokens] = 1
    return (
        token,
        torch.full((1,), prompt_tokens, dtype=torch.int32),
        torch.full((1,), speech_tokens, dtype=torch.int32),
        torch.randn(1, 192, dtype=dtype),
    )


def make_spec(token_bucket: int) -> GraphSpec:
    return GraphSpec(
        name=token_to_mu_graph_name(token_bucket),
        filename=token_to_mu_filename(token_bucket),
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        make_module=make_module,
        make_dummy_inputs=lambda token_bucket=token_bucket: make_dummy_inputs(
            token_bucket
        ),
        input_dtypes={
            "token": "int32",
            "prompt_token_len": "int32",
            "speech_token_len": "int32",
            "embedding": "float32",
        },
        output_dtypes={
            "mu": "float32",
            "mask": "float32",
            "spks": "float32",
            "prompt_mel_len": "int32",
            "output_mel_len": "int32",
        },
    )


TOKEN_TO_MU_BUCKET_SPECS = tuple(
    make_spec(bucket) for bucket in TOKEN_TO_MU_TOKEN_BUCKETS
)
TOKEN_TO_MU_SPEC = TOKEN_TO_MU_BUCKET_SPECS[0]
