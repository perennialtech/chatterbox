from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence


@dataclass
class ModelConfig:
    n_mels: int = 128
    n_audio_ctx: int = 1500
    n_audio_state: int = 1280
    n_audio_head: int = 20
    n_audio_layer: int = 6
    n_codebook_size: int = 3**8

    use_sdpa: bool = False


class LayerNorm(nn.LayerNorm):
    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float() if self.bias is not None else None,
            self.eps,
        ).to(dtype=x.dtype)


class Linear(nn.Linear):
    def forward(self, x: Tensor) -> Tensor:
        return F.linear(
            x,
            self.weight.to(dtype=x.dtype),
            None if self.bias is None else self.bias.to(dtype=x.dtype),
        )


class Conv1d(nn.Conv1d):
    def _conv_forward(
        self,
        x: Tensor,
        weight: Tensor,
        bias: Optional[Tensor],
    ) -> Tensor:
        return super()._conv_forward(
            x,
            weight.to(dtype=x.dtype),
            None if bias is None else bias.to(dtype=x.dtype),
        )


def pad_mel_batch(data: List[Tensor]) -> Tuple[Tensor, Tensor]:
    lengths = torch.tensor([x.size(1) for x in data], dtype=torch.int64)
    padded = pad_sequence([x.t() for x in data], batch_first=True, padding_value=0)
    return padded.transpose(1, 2), lengths


def make_non_pad_mask(lengths: Tensor, max_len: Optional[int] = None) -> Tensor:
    if max_len is None:
        max_len = int(lengths.max().item())

    seq_range = torch.arange(max_len, dtype=torch.int64, device=lengths.device)
    seq_range = seq_range.unsqueeze(0).expand(lengths.size(0), max_len)
    return seq_range < lengths.unsqueeze(-1)


def mask_to_bias(mask: Tensor, dtype: torch.dtype) -> Tensor:
    return (1.0 - mask.to(dtype=dtype)) * -1.0e10


def precompute_rotary_frequencies(
    dim: int,
    end: int,
    theta: float = 10000.0,
) -> Tuple[Tensor, Tensor]:
    assert dim % 2 == 0

    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    freqs = torch.cat((freqs, freqs), dim=-1)
    return freqs.cos(), freqs.sin()


def apply_rotary_emb(
    q: Tensor,
    k: Tensor,
    freqs_cos: Tensor,
    freqs_sin: Tensor,
) -> Tuple[Tensor, Tensor]:
    if freqs_cos.dtype != q.dtype:
        freqs_cos = freqs_cos.to(dtype=q.dtype)
        freqs_sin = freqs_sin.to(dtype=q.dtype)

    cos = freqs_cos.unsqueeze(0).unsqueeze(2)
    sin = freqs_sin.unsqueeze(0).unsqueeze(2)

    q_left, q_right = q[..., : q.shape[-1] // 2], q[..., q.shape[-1] // 2 :]
    k_left, k_right = k[..., : k.shape[-1] // 2], k[..., k.shape[-1] // 2 :]

    q_rot = torch.cat((-q_right, q_left), dim=-1)
    k_rot = torch.cat((-k_right, k_left), dim=-1)

    return q * cos + q_rot * sin, k * cos + k_rot * sin


class FSQCodebook(nn.Module):
    def __init__(self, dim: int, level: int = 3):
        super().__init__()
        self.project_down = Linear(dim, 8)
        self.level = level
        self.embed = None
        self.register_buffer(
            "powers",
            torch.tensor([level**i for i in range(8)], dtype=torch.int64),
            persistent=False,
        )

    def encode(self, x: Tensor) -> Tensor:
        x_shape = x.shape
        x = x.reshape(-1, x_shape[-1])

        h = self.project_down(x).float()
        h = h.tanh()
        h = h * 0.9990000128746033
        h = h.round() + 1.0

        powers = self.powers.to(dtype=h.dtype).unsqueeze(0)
        indices = torch.sum(h * powers, dim=-1)
        return indices.reshape(*x_shape[:-1]).to(dtype=torch.int32)

    def decode(self, embed_ind: Tensor) -> Tensor:
        raise NotImplementedError("FSQ decoding requires the unavailable up-projector")


class FSQVectorQuantization(nn.Module):
    def __init__(self, dim: int, codebook_size: int):
        super().__init__()
        if codebook_size != 3**8:
            raise ValueError("S3Tokenizer v2 FSQ requires a 3**8 codebook")
        self._codebook = FSQCodebook(dim=dim, level=3)
        self.codebook_size = codebook_size

    @property
    def codebook(self):
        return self._codebook.embed

    def encode(self, x: Tensor) -> Tensor:
        return self._codebook.encode(x)

    def decode(self, embed_ind: Tensor) -> Tensor:
        return self._codebook.decode(embed_ind).transpose(1, 2)


class FSMNMultiHeadAttention(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
        use_sdpa: bool = False,
    ):
        super().__init__()
        if n_state % n_head != 0:
            raise ValueError("n_state must be divisible by n_head")

        self.n_state = n_state
        self.n_head = n_head
        self.head_dim = n_state // n_head
        self.scale = self.head_dim**-0.25
        self.use_sdpa = use_sdpa

        self.query = Linear(n_state, n_state)
        self.key = Linear(n_state, n_state, bias=False)
        self.value = Linear(n_state, n_state)
        self.out = Linear(n_state, n_state)

        self.fsmn_block = nn.Conv1d(
            n_state,
            n_state,
            kernel_size,
            stride=1,
            padding=0,
            groups=n_state,
            bias=False,
        )
        left_padding = (kernel_size - 1) // 2
        right_padding = kernel_size - 1 - left_padding
        self.pad_fn = nn.ConstantPad1d((left_padding, right_padding), 0.0)

    def forward_fsmn(self, value: Tensor, mask_pad: Optional[Tensor]) -> Tensor:
        value = value.reshape(value.shape[0], value.shape[1], self.n_state)

        if mask_pad is not None:
            value = value * mask_pad.to(dtype=value.dtype)

        memory = self.pad_fn(value.transpose(1, 2))
        memory = self.fsmn_block(memory).transpose(1, 2)
        memory = memory + value

        if mask_pad is not None:
            memory = memory * mask_pad.to(dtype=memory.dtype)

        return memory

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        mask_pad: Optional[Tensor] = None,
        freqs_cos: Optional[Tensor] = None,
        freqs_sin: Optional[Tensor] = None,
    ) -> Tensor:
        q = self.query(x).reshape(x.shape[0], x.shape[1], self.n_head, self.head_dim)
        k = self.key(x).reshape(x.shape[0], x.shape[1], self.n_head, self.head_dim)
        v = self.value(x).reshape(x.shape[0], x.shape[1], self.n_head, self.head_dim)

        if freqs_cos is not None and freqs_sin is not None:
            q, k = apply_rotary_emb(q, k, freqs_cos, freqs_sin)

        fsm_memory = self.forward_fsmn(v, mask_pad)

        q = q.permute(0, 2, 1, 3) * self.scale
        k = k.permute(0, 2, 1, 3) * self.scale
        v = v.permute(0, 2, 1, 3)

        if self.use_sdpa:
            if mask is None:
                raise ValueError("SDPA requires an attention mask")
            attended = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=mask,
                dropout_p=0.0,
                scale=1.0,
            )
        else:
            scores = q @ k.transpose(-2, -1)
            if mask is not None:
                scores = scores + mask
            weights = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
            attended = weights @ v

        attended = attended.permute(0, 2, 1, 3).flatten(start_dim=2)
        return self.out(attended) + fsm_memory


class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        n_state: int,
        n_head: int,
        kernel_size: int = 31,
        use_sdpa: bool = False,
    ):
        super().__init__()
        self.attn = FSMNMultiHeadAttention(
            n_state,
            n_head,
            kernel_size=kernel_size,
            use_sdpa=use_sdpa,
        )
        self.attn_ln = LayerNorm(n_state, eps=1e-5)

        self.mlp = nn.Sequential(
            Linear(n_state, n_state * 4),
            nn.GELU(),
            Linear(n_state * 4, n_state),
        )
        self.mlp_ln = LayerNorm(n_state)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        mask_pad: Optional[Tensor] = None,
        freqs_cos: Optional[Tensor] = None,
        freqs_sin: Optional[Tensor] = None,
    ) -> Tensor:
        x = x + self.attn(
            self.attn_ln(x),
            mask=mask,
            mask_pad=mask_pad,
            freqs_cos=freqs_cos,
            freqs_sin=freqs_sin,
        )
        return x + self.mlp(self.mlp_ln(x))


class AudioEncoderV2(nn.Module):
    def __init__(
        self,
        n_mels: int,
        n_ctx: int,
        n_state: int,
        n_head: int,
        n_layer: int,
        stride: int,
        use_sdpa: bool,
    ):
        super().__init__()
        self.stride = stride

        self.conv1 = Conv1d(n_mels, n_state, kernel_size=3, stride=stride, padding=1)
        self.conv2 = Conv1d(n_state, n_state, kernel_size=3, stride=2, padding=1)

        freqs_cos, freqs_sin = precompute_rotary_frequencies(
            n_state // n_head,
            max(n_ctx, 2048),
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        self.blocks = nn.ModuleList(
            [
                ResidualAttentionBlock(n_state, n_head, use_sdpa=use_sdpa)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x: Tensor, x_len: Tensor) -> Tuple[Tensor, Tensor]:
        x_len = x_len.to(dtype=torch.int64)

        t = x.shape[-1]
        mask = make_non_pad_mask(x_len, t).unsqueeze(1)
        x = F.gelu(self.conv1(x * mask.to(dtype=x.dtype)))

        x_len = (x_len + 2 - (3 - 1) - 1) // self.stride + 1
        t = (t + 2 - (3 - 1) - 1) // self.stride + 1
        mask = make_non_pad_mask(x_len, t).unsqueeze(1)
        x = F.gelu(self.conv2(x * mask.to(dtype=x.dtype)))

        x_len = (x_len + 2 - (3 - 1) - 1) // 2 + 1
        t = (t + 2 - (3 - 1) - 1) // 2 + 1
        mask = make_non_pad_mask(x_len, t).unsqueeze(1)

        x = x.permute(0, 2, 1)

        freqs_cos = self.freqs_cos[: x.shape[1]]
        freqs_sin = self.freqs_sin[: x.shape[1]]
        mask_pad = mask.transpose(1, 2)
        attention_bias = mask_to_bias(mask, x.dtype).unsqueeze(1)

        for block in self.blocks:
            x = block(x, attention_bias, mask_pad, freqs_cos, freqs_sin)

        return x, x_len


class S3TokenizerV2(nn.Module):
    def __init__(
        self,
        name: str,
        config: Optional[ModelConfig] = None,
    ):
        super().__init__()
        if "v2" not in name:
            raise ValueError("S3TokenizerV2 requires a v2 model name")

        self.name = name
        self.config = config if config is not None else ModelConfig()

        self.encoder = AudioEncoderV2(
            self.config.n_mels,
            self.config.n_audio_ctx,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2,
            self.config.use_sdpa,
        )
        self.quantizer = FSQVectorQuantization(
            self.config.n_audio_state,
            self.config.n_codebook_size,
        )

    def forward(self, mel: Tensor, mel_len: Tensor) -> Tuple[Tensor, Tensor]:
        return self.quantize(mel, mel_len)

    def quantize(self, mel: Tensor, mel_len: Tensor) -> Tuple[Tensor, Tensor]:
        hidden, code_len = self.encoder(mel, mel_len)
        return self.quantizer.encode(hidden), code_len

    @property
    def device(self):
        return next(self.parameters()).device

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
