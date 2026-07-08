import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d, ConvTranspose1d, Parameter
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

from .source import DeterministicSourceModuleHnNSF
from .stft import RealISTFT, RealSTFT


class Snake(nn.Module):
    def __init__(
        self,
        in_features: int,
        alpha: float = 1.0,
        alpha_trainable: bool = True,
        alpha_logscale: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.alpha_logscale = alpha_logscale
        self.no_div_by_zero = 1e-9

        if alpha_logscale:
            self.alpha = Parameter(torch.zeros(in_features) * alpha)
        else:
            self.alpha = Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        alpha = self.alpha.reshape(1, -1, 1)
        if alpha.dtype != x.dtype:
            alpha = alpha.to(dtype=x.dtype)

        if self.alpha_logscale:
            alpha = torch.exp(alpha)

        y = torch.sin(x * alpha)
        return x + y.square() * torch.reciprocal(alpha + self.no_div_by_zero)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return int((kernel_size * dilation - dilation) / 2)


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01) -> None:
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def _remove_weight_parametrization(module: nn.Module) -> None:
    if parametrize.is_parametrized(module, "weight"):
        parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        channels: int = 512,
        kernel_size: int = 3,
        dilations: Optional[Sequence[int]] = None,
    ):
        super().__init__()

        if dilations is None:
            dilations = (1, 3, 5)

        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()

        for dilation in dilations:
            self.convs1.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=dilation,
                        padding=get_padding(kernel_size, dilation),
                    )
                )
            )
            self.convs2.append(
                weight_norm(
                    Conv1d(
                        channels,
                        channels,
                        kernel_size,
                        1,
                        dilation=1,
                        padding=get_padding(kernel_size, 1),
                    )
                )
            )

        self.convs1.apply(init_weights)
        self.convs2.apply(init_weights)

        self.activations1 = nn.ModuleList(
            [Snake(channels, alpha_logscale=False) for _ in range(len(self.convs1))]
        )
        self.activations2 = nn.ModuleList(
            [Snake(channels, alpha_logscale=False) for _ in range(len(self.convs2))]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for idx in range(len(self.convs1)):
            xt = self.activations1[idx](x)
            xt = self.convs1[idx](xt)
            xt = self.activations2[idx](xt)
            xt = self.convs2[idx](xt)
            x = xt + x
        return x


class HiFTGenerator(nn.Module):
    def __init__(
        self,
        in_channels: int = 80,
        base_channels: int = 512,
        nb_harmonics: int = 8,
        sampling_rate: int = 22050,
        nsf_alpha: float = 0.1,
        nsf_sigma: float = 0.003,
        nsf_voiced_threshold: float = 10,
        upsample_rates: Optional[Sequence[int]] = None,
        upsample_kernel_sizes: Optional[Sequence[int]] = None,
        istft_params: Optional[Dict[str, int]] = None,
        resblock_kernel_sizes: Optional[Sequence[int]] = None,
        resblock_dilation_sizes: Optional[Sequence[Sequence[int]]] = None,
        source_resblock_kernel_sizes: Optional[Sequence[int]] = None,
        source_resblock_dilation_sizes: Optional[Sequence[Sequence[int]]] = None,
        lrelu_slope: float = 0.1,
        audio_limit: float = 0.99,
        f0_predictor: Optional[torch.nn.Module] = None,
    ):
        super().__init__()

        if upsample_rates is None:
            upsample_rates = (8, 8)
        else:
            upsample_rates = tuple(upsample_rates)

        if upsample_kernel_sizes is None:
            upsample_kernel_sizes = (16, 16)
        else:
            upsample_kernel_sizes = tuple(upsample_kernel_sizes)

        if istft_params is None:
            istft_params = {"n_fft": 16, "hop_len": 4}
        else:
            istft_params = dict(istft_params)

        if resblock_kernel_sizes is None:
            resblock_kernel_sizes = (3, 7, 11)
        else:
            resblock_kernel_sizes = tuple(resblock_kernel_sizes)

        if resblock_dilation_sizes is None:
            resblock_dilation_sizes = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
        else:
            resblock_dilation_sizes = tuple(tuple(d) for d in resblock_dilation_sizes)

        if source_resblock_kernel_sizes is None:
            source_resblock_kernel_sizes = (7, 11)
        else:
            source_resblock_kernel_sizes = tuple(source_resblock_kernel_sizes)

        if source_resblock_dilation_sizes is None:
            source_resblock_dilation_sizes = ((1, 3, 5), (1, 3, 5))
        else:
            source_resblock_dilation_sizes = tuple(
                tuple(d) for d in source_resblock_dilation_sizes
            )

        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise ValueError(
                "upsample_rates and upsample_kernel_sizes must have the same length: "
                f"{len(upsample_rates)} != {len(upsample_kernel_sizes)}"
            )

        if len(resblock_kernel_sizes) != len(resblock_dilation_sizes):
            raise ValueError(
                "resblock_kernel_sizes and resblock_dilation_sizes must have the "
                "same length: "
                f"{len(resblock_kernel_sizes)} != {len(resblock_dilation_sizes)}"
            )

        if len(source_resblock_kernel_sizes) != len(upsample_rates):
            raise ValueError(
                "source_resblock_kernel_sizes length must match upsample_rates "
                "length: "
                f"{len(source_resblock_kernel_sizes)} != {len(upsample_rates)}"
            )

        if len(source_resblock_dilation_sizes) != len(upsample_rates):
            raise ValueError(
                "source_resblock_dilation_sizes length must match upsample_rates "
                "length: "
                f"{len(source_resblock_dilation_sizes)} != {len(upsample_rates)}"
            )

        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            if k < u:
                raise ValueError(
                    f"upsample_kernel_sizes[{i}] must be >= upsample_rates[{i}]: "
                    f"{k} < {u}"
                )
            if (k - u) % 2 != 0:
                raise ValueError(
                    f"upsample_kernel_sizes[{i}] - upsample_rates[{i}] must be even: "
                    f"{k} - {u}"
                )

        self.out_channels = 1
        self.nb_harmonics = nb_harmonics
        self.sampling_rate = sampling_rate
        self.istft_params = istft_params
        self.n_fft = istft_params["n_fft"]
        self.hop_len = istft_params["hop_len"]
        self.freq_bins = self.n_fft // 2 + 1
        self.stft_channels = self.freq_bins * 2
        self.max_log_magnitude = math.log(1e2)
        self.lrelu_slope = lrelu_slope
        self.audio_limit = audio_limit

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.source_hop = math.prod(upsample_rates) * self.hop_len

        self.m_source = DeterministicSourceModuleHnNSF(
            sampling_rate=sampling_rate,
            harmonic_num=nb_harmonics,
            sine_amp=nsf_alpha,
            add_noise_std=nsf_sigma,
            voiced_threshold=nsf_voiced_threshold,
        )

        self.conv_pre = weight_norm(Conv1d(in_channels, base_channels, 7, 1, padding=3))

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        base_channels // (2**i),
                        base_channels // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.source_downs = nn.ModuleList()
        self.source_resblocks = nn.ModuleList()

        downsample_rates = [1] + list(reversed(upsample_rates))[:-1]
        downsample_cum_rates = []
        cumulative_rate = 1
        for rate in downsample_rates:
            cumulative_rate *= rate
            downsample_cum_rates.append(cumulative_rate)

        for i, (u, k, d) in enumerate(
            zip(
                reversed(downsample_cum_rates),
                source_resblock_kernel_sizes,
                source_resblock_dilation_sizes,
            )
        ):
            channels = base_channels // (2 ** (i + 1))
            if u == 1:
                self.source_downs.append(Conv1d(self.stft_channels, channels, 1, 1))
            else:
                self.source_downs.append(
                    Conv1d(
                        self.stft_channels,
                        channels,
                        u * 2,
                        u,
                        padding=u // 2,
                    )
                )

            self.source_resblocks.append(ResBlock(channels, k, d))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = base_channels // (2 ** (i + 1))
            for k, d in zip(resblock_kernel_sizes, resblock_dilation_sizes):
                self.resblocks.append(ResBlock(ch, k, d))

        self.conv_post = weight_norm(Conv1d(ch, self.stft_channels, 7, 1, padding=3))

        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

        self.reflection_pad = nn.ReflectionPad1d((1, 0))

        self.real_stft = RealSTFT(self.n_fft, self.hop_len, center=True)
        self.real_istft = RealISTFT(self.n_fft, self.hop_len, center=True)

        self.f0_predictor = f0_predictor
        self._compiled_decode_from_source = None

    def remove_weight_norm(self) -> None:
        for module in list(self.modules()):
            _remove_weight_parametrization(module)

    def optimize_for_inference(self) -> "HiFTGenerator":
        self.eval()
        self.remove_weight_norm()
        return self

    def compile_for_inference(self) -> "HiFTGenerator":
        self.optimize_for_inference()
        if getattr(self, "_compiled_for_inference", False):
            return self

        self._compiled_decode_from_source = torch.compile(
            self._decode_from_source_impl,
            mode="default",
            backend="inductor",
            dynamic=True,
        )
        self._compiled_for_inference = True
        return self

    def _stft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dtype not in (torch.float32, torch.float64):
            x = x.float()
        return self.real_stft(x)

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        magnitude = magnitude.clamp_max(1e2)
        if magnitude.dtype not in (torch.float32, torch.float64):
            magnitude = magnitude.float()
        if phase.dtype != magnitude.dtype:
            phase = phase.to(dtype=magnitude.dtype)
        return self.real_istft(magnitude, phase)

    def predict_f0(self, speech_feat: torch.Tensor) -> torch.Tensor:
        return self.f0_predictor(speech_feat.contiguous())

    def source_from_f0(
        self,
        f0: torch.Tensor,
        source_phase: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> torch.Tensor:
        source_f0 = f0[:, None].repeat_interleave(self.source_hop, dim=2)
        source, _ = self.m_source(source_f0, source_phase, source_noise)
        return source

    def _source_stft(self, s: torch.Tensor) -> torch.Tensor:
        s_stft_real, s_stft_imag = self._stft(s.squeeze(1))
        return torch.cat((s_stft_real, s_stft_imag), dim=1)

    def _upsample_stage(self, x: torch.Tensor, i: int) -> torch.Tensor:
        x = F.leaky_relu(x, self.lrelu_slope)
        x = self.ups[i](x)
        if i == self.num_upsamples - 1:
            x = self.reflection_pad(x)
        return x

    def _source_fusion_stage(
        self, x: torch.Tensor, s_stft: torch.Tensor, i: int
    ) -> torch.Tensor:
        si = self.source_downs[i](s_stft)
        si = self.source_resblocks[i](si)
        if si.size(-1) != x.size(-1):
            raise ValueError(
                "source fusion length mismatch at upsample stage "
                f"{i}: x length {x.size(-1)}, source length {si.size(-1)}"
            )
        return x + si

    def _resblock_stage(self, x: torch.Tensor, i: int) -> torch.Tensor:
        offset = i * self.num_kernels
        xs = self.resblocks[offset](x)
        for j in range(1, self.num_kernels):
            xs = xs + self.resblocks[offset + j](x)
        return xs / self.num_kernels

    def _projection(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        log_magnitude = x[:, : self.freq_bins, :].clamp_max(self.max_log_magnitude)
        magnitude = torch.exp(log_magnitude)
        phase = torch.sin(x[:, self.freq_bins :, :])
        return magnitude, phase

    def _forward_features(
        self, x: torch.Tensor, s_stft: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_pre(x)
        for i in range(self.num_upsamples):
            x = self._upsample_stage(x, i)
            x = self._source_fusion_stage(x, s_stft, i)
            x = self._resblock_stage(x, i)
        magnitude, phase = self._projection(x)
        return magnitude, phase

    def _decode_from_source_impl(
        self, speech_feat: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        s_stft = self._source_stft(source)
        s_stft = s_stft.to(device=speech_feat.device, dtype=speech_feat.dtype)
        magnitude, phase = self._forward_features(speech_feat.contiguous(), s_stft)
        x = self._istft(magnitude, phase)
        x.clamp_(-self.audio_limit, self.audio_limit)
        return x

    def decode_from_source(
        self, speech_feat: torch.Tensor, source: torch.Tensor
    ) -> torch.Tensor:
        compiled_decode = self._compiled_decode_from_source
        if compiled_decode is not None:
            return compiled_decode(speech_feat, source)
        return self._decode_from_source_impl(speech_feat, source)

    def forward(
        self,
        speech_feat: torch.Tensor,
        source_phase: torch.Tensor,
        source_noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        f0 = self.predict_f0(speech_feat)
        source = self.source_from_f0(f0, source_phase, source_noise)
        generated_speech = self.decode_from_source(speech_feat, source)
        return generated_speech, source

    @torch.inference_mode()
    def inference(
        self,
        speech_feat: torch.Tensor,
        cache_source: Optional[torch.Tensor] = None,
        source_phase: Optional[torch.Tensor] = None,
        source_noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        speech_feat = speech_feat.contiguous()
        f0 = self.predict_f0(speech_feat)

        if source_phase is None:
            source_phase = torch.empty(
                f0.size(0),
                self.nb_harmonics + 1,
                1,
                device=f0.device,
                dtype=f0.dtype,
            )
            source_phase.uniform_(-math.pi, math.pi)
            source_phase[:, :1, :] = 0.0

        source_f0 = f0[:, None].repeat_interleave(self.source_hop, dim=2)

        if source_noise is None:
            source_noise = torch.randn_like(
                source_f0.expand(-1, self.nb_harmonics + 1, -1)
            )

        s = self.source_from_f0(
            f0, source_phase=source_phase, source_noise=source_noise
        )

        if cache_source is not None and cache_source.numel() != 0:
            if cache_source.ndim != 3:
                raise ValueError(
                    f"cache_source must have shape [B, C, T], got {cache_source.shape}"
                )
            if cache_source.size(0) != s.size(0) or cache_source.size(1) != s.size(1):
                raise ValueError(
                    "cache_source shape mismatch: expected batch/channel "
                    f"({s.size(0)}, {s.size(1)}), got "
                    f"({cache_source.size(0)}, {cache_source.size(1)})"
                )
            n = min(s.size(2), cache_source.size(2))
            s[:, :, :n].copy_(cache_source[:, :, :n].to(device=s.device, dtype=s.dtype))

        generated_speech = self.decode_from_source(speech_feat, s)
        return generated_speech, s
