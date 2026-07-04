import torch
import torch.nn as nn

from .utils.mask import make_pad_mask


class S3TokenEncoder(nn.Module):
    def __init__(
        self,
        input_embedding: nn.Embedding,
        encoder: nn.Module,
        encoder_proj: nn.Linear,
        vocab_size: int,
    ):
        super().__init__()
        self.input_embedding = input_embedding
        self.encoder = encoder
        self.encoder_proj = encoder_proj
        self.vocab_size = vocab_size

    def forward(self, token: torch.Tensor, token_len: torch.Tensor):
        token = token.long().clamp(min=0, max=self.vocab_size - 1)
        token_mask = (~make_pad_mask(token_len, max_len=token.size(1))).unsqueeze(-1)
        embedded = self.input_embedding(token) * token_mask.to(
            dtype=self.input_embedding.weight.dtype
        )
        h, h_mask = self.encoder(embedded, token_len)
        mu = self.encoder_proj(h).transpose(1, 2).contiguous()
        mask = h_mask.to(dtype=mu.dtype)
        return mu, mask
