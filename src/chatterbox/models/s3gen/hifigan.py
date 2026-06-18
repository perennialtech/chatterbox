# jrm: adapted from CosyVoice/cosyvoice/hifigan/generator.py
#      most modules should be reusable, but I found their SineGen changed a git.

# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Kai Hu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HIFI-GAN"""

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv1d
from torch.nn import ConvTranspose1d
from torch.nn import Parameter
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

from ...timing import ensure_timer


class Snake(nn.Module):
    """
    Implementation of a sine-based periodic activation function
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter
    References:
        - This activation function is from this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = Snake(256)
        >>> x = torch.randn(1, 256, 128)
        >>> x = a1(x)
    """

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
        """
        Snake := x + 1 / alpha * sin^2(x * alpha)
        """
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


"""hifigan based generator implementation.

This code is modified from https://github.com/jik876/hifi-gan
,https://github.com/kan-bayashi/ParallelWaveGAN and
https://github.com/NVIDIA/BigVGAN

"""


class ResBlock(torch.nn.Module):
    """Residual block module in HiFiGAN/BigVGAN."""

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

    def remove_weight_norm(self) -> None:
        for conv in self.convs1:
            _remove_weight_parametrization(conv)
        for conv in self.convs2:
            _remove_weight_parametrization(conv)


class SineGen(torch.nn.Module):
    """Definition of sine generator.

    SineGen(
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
    )

    Input:
        f0: [B, 1, T], Hz

    Output:
        sine_waves: [B, harmonic_num + 1, T]
        uv: [B, 1, T]
    """

    def __init__(
        self,
        samp_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold

        self.register_buffer(
            "harmonic_factors",
            torch.arange(1, harmonic_num + 2, dtype=torch.float32).reshape(1, -1, 1),
            persistent=False,
        )

    def _f02uv(self, f0: torch.Tensor) -> torch.Tensor:
        return (f0 > self.voiced_threshold).to(dtype=f0.dtype)

    @torch.no_grad()
    def forward(self, f0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param f0: [B, 1, T], Hz
        :return: sine_waves [B, harmonic_num + 1, T], uv [B, 1, T]
        """
        if f0.dtype != torch.float32:
            f0 = f0.float()

        base_cycles = torch.cumsum(f0 / self.sampling_rate, dim=-1)
        theta = torch.remainder(base_cycles * self.harmonic_factors, 1.0)
        theta.mul_(2.0 * math.pi)

        phase_vec = torch.empty(
            f0.size(0),
            self.harmonic_num + 1,
            1,
            device=f0.device,
            dtype=f0.dtype,
        )
        phase_vec.uniform_(-math.pi, math.pi)
        phase_vec[:, :1, :] = 0.0

        theta.add_(phase_vec)
        sine_waves = torch.sin(theta)
        sine_waves.mul_(self.sine_amp)

        uv = self._f02uv(f0)

        noise_amp = uv * self.noise_std + (1.0 - uv) * (self.sine_amp / 3.0)
        noise = torch.randn_like(sine_waves)
        noise.mul_(noise_amp)

        sine_waves.mul_(uv)
        sine_waves.add_(noise)

        return sine_waves, uv


class SourceModuleHnNSF(torch.nn.Module):
    """SourceModule for hn-nsf.

    Input:
        f0: [B, 1, T]

    Output:
        source: [B, 1, T]
        uv: [B, 1, T]
    """

    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        self.l_sin_gen = SineGen(
            sampling_rate,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
        )

        self.l_linear = torch.nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch.nn.Tanh()

    def forward(self, f0: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sine_wavs, uv = self.l_sin_gen(f0)

        source = self.l_linear(sine_wavs.transpose(1, 2))
        source = self.l_tanh(source)
        source = source.transpose(1, 2)

        return source, uv


class HiFTGenerator(nn.Module):
    """
    HiFTNet Generator: Neural Source Filter + ISTFTNet
    https://arxiv.org/abs/2309.09493
    """

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

        if len(upsample_rates) == 0:
            raise ValueError("upsample_rates must not be empty")
        if len(upsample_rates) != len(upsample_kernel_sizes):
            raise ValueError("upsample_rates and upsample_kernel_sizes must match")
        if len(resblock_kernel_sizes) == 0:
            raise ValueError("resblock_kernel_sizes must not be empty")
        if len(resblock_kernel_sizes) != len(resblock_dilation_sizes):
            raise ValueError(
                "resblock_kernel_sizes and resblock_dilation_sizes must match"
            )
        if len(source_resblock_kernel_sizes) != len(source_resblock_dilation_sizes):
            raise ValueError(
                "source_resblock_kernel_sizes and source_resblock_dilation_sizes must match"
            )
        if len(source_resblock_kernel_sizes) != len(upsample_rates):
            raise ValueError(
                "source resblock configuration must have one entry per upsample stage"
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

        self.m_source = SourceModuleHnNSF(
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

        self.register_buffer(
            "stft_window",
            torch.hann_window(self.n_fft, periodic=True, dtype=torch.float32),
            persistent=False,
        )

        self.f0_predictor = f0_predictor

    def remove_weight_norm(self) -> None:
        for module in list(self.modules()):
            _remove_weight_parametrization(module)

    def optimize_for_inference(self) -> "HiFTGenerator":
        self.eval()
        self.remove_weight_norm()
        return self

    def compile_for_inference(self) -> "HiFTGenerator":
        self._decode_fast = torch.compile(
            self._decode_fast,
            mode="default", # other modes are broken
            dynamic=True,
        )
        return self

    def _stft(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dtype not in (torch.float32, torch.float64):
            x = x.float()

        window = self.stft_window.to(device=x.device, dtype=x.dtype)

        spec = torch.stft(
            x,
            self.n_fft,
            self.hop_len,
            self.n_fft,
            window=window,
            return_complex=True,
        )
        spec = torch.view_as_real(spec)
        return spec[..., 0], spec[..., 1]

    def _istft(self, magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        magnitude = magnitude.clamp_max(1e2)

        if magnitude.dtype not in (torch.float32, torch.float64):
            magnitude = magnitude.float()
        if phase.dtype != magnitude.dtype:
            phase = phase.to(dtype=magnitude.dtype)

        window = self.stft_window.to(device=magnitude.device, dtype=magnitude.dtype)
        spec = torch.polar(magnitude, phase)

        return torch.istft(
            spec,
            self.n_fft,
            self.hop_len,
            self.n_fft,
            window=window,
        )

    def _source_from_f0(self, f0: torch.Tensor) -> torch.Tensor:
        source_f0 = f0[:, None].repeat_interleave(self.source_hop, dim=2)
        source, _ = self.m_source(source_f0)
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
        self,
        x: torch.Tensor,
        s_stft: torch.Tensor,
        i: int,
    ) -> torch.Tensor:
        si = self.source_downs[i](s_stft)
        si = self.source_resblocks[i](si)
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

    def _decode_fast(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s_stft = self._source_stft(s)
        x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            x = self._upsample_stage(x, i)
            x = self._source_fusion_stage(x, s_stft, i)
            x = self._resblock_stage(x, i)

        magnitude, phase = self._projection(x)
        x = self._istft(magnitude, phase)
        x.clamp_(-self.audio_limit, self.audio_limit)

        return x

    def _decode_timed(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        timer,
    ) -> torch.Tensor:
        with timer.track("decode.source_stft"):
            s_stft = self._source_stft(s)

        with timer.track("decode.conv_pre"):
            x = self.conv_pre(x)

        for i in range(self.num_upsamples):
            with timer.track(f"decode.upsample_{i}"):
                x = self._upsample_stage(x, i)

            with timer.track(f"decode.source_fusion_{i}"):
                x = self._source_fusion_stage(x, s_stft, i)

            with timer.track(f"decode.resblocks_{i}"):
                x = self._resblock_stage(x, i)

        with timer.track("decode.projection"):
            magnitude, phase = self._projection(x)

        with timer.track("decode.istft"):
            x = self._istft(magnitude, phase)

        with timer.track("decode.clamp"):
            x.clamp_(-self.audio_limit, self.audio_limit)

        return x

    def decode(
        self,
        x: torch.Tensor,
        s: torch.Tensor,
        timer=None,
    ) -> torch.Tensor:
        timer = ensure_timer(timer, x.device)
        with timer.track("decode_fast"):
            return self._decode_fast(x, s)

    def forward(
        self,
        batch: dict,
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        speech_feat = (
            batch["speech_feat"]
            .to(device, non_blocking=True)
            .transpose(1, 2)
            .contiguous()
        )

        f0 = self.f0_predictor(speech_feat)
        s = self._source_from_f0(f0)

        generated_speech = self.decode(x=speech_feat, s=s)
        return generated_speech, f0

    @torch.inference_mode()
    def inference(
        self,
        speech_feat: torch.Tensor,
        cache_source: Optional[torch.Tensor] = None,
        timer=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        speech_feat = speech_feat.contiguous()

        timer = ensure_timer(timer, speech_feat.device)

        with timer.track("f0_predictor"):
            f0 = self.f0_predictor(speech_feat)

        with timer.track("source_upsample"):
            s = f0[:, None].repeat_interleave(self.source_hop, dim=2)

        with timer.track("source_module"):
            s, _ = self.m_source(s)

        if cache_source is not None and cache_source.numel() != 0:
            with timer.track("cache_source"):
                s[:, :, : cache_source.shape[2]].copy_(cache_source)

        with timer.track("decode"):
            generated_speech = self._decode_fast(speech_feat, s)

        return generated_speech, s
