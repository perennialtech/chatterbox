import math
from typing import Optional

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
    ):
        super().__init__()
        inner = dim * mult
        if activation_fn in {"gelu", "geglu", "gelu-approximate", "geglu-approximate"}:
            activation = nn.GELU()
        elif activation_fn in {"silu", "swish"}:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        self.net = nn.Sequential(
            nn.Linear(dim, inner),
            activation,
            nn.Dropout(dropout),
            nn.Linear(inner, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        attention_bias: bool = False,
    ):
        super().__init__()
        self.heads = num_attention_heads
        self.head_dim = attention_head_dim
        self.inner_dim = num_attention_heads * attention_head_dim
        self.to_q = nn.Linear(dim, self.inner_dim, bias=attention_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias=attention_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias=attention_bias)
        self.to_out = nn.Sequential(nn.Linear(self.inner_dim, dim), nn.Dropout(dropout))

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        return x.view(bsz, seq, self.heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        q = self._heads(self.to_q(hidden_states))
        k = self._heads(self.to_k(hidden_states))
        v = self._heads(self.to_v(hidden_states))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.ndim == 3:
                attention_mask = attention_mask.unsqueeze(1)
            scores = scores + attention_mask.to(dtype=scores.dtype)

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(hidden_states.size(0), hidden_states.size(1), self.inner_dim)
        )
        return self.to_out(out)


class BasicTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu",
        attention_bias: bool = False,
        norm_elementwise_affine: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.attn1 = SelfAttention(
            dim=dim,
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            dropout=dropout,
            attention_bias=attention_bias,
        )
        self.norm3 = nn.LayerNorm(dim, elementwise_affine=norm_elementwise_affine)
        self.ff = FeedForward(
            dim=dim,
            dropout=dropout,
            activation_fn=activation_fn,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn1(
            self.norm1(hidden_states), attention_mask
        )

        norm_hidden_states = self.norm3(hidden_states)
        ff_output = self.ff(norm_hidden_states)

        return hidden_states + ff_output
