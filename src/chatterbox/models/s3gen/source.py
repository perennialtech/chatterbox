import math

import torch
import torch.nn as nn


class DeterministicSineGen(nn.Module):
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

    def forward(
        self,
        f0: torch.Tensor,
        phase: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f0 = f0.float()
        base_cycles = torch.cumsum(f0 / self.sampling_rate, dim=-1)
        theta = torch.remainder(base_cycles * self.harmonic_factors.to(f0.device), 1.0)
        theta = theta * (2.0 * math.pi) + phase.to(device=f0.device, dtype=f0.dtype)

        sine_waves = torch.sin(theta) * self.sine_amp
        uv = (f0 > self.voiced_threshold).to(dtype=f0.dtype)
        noise_amp = uv * self.noise_std + (1.0 - uv) * (self.sine_amp / 3.0)
        sine_waves = (
            sine_waves * uv + noise.to(device=f0.device, dtype=f0.dtype) * noise_amp
        )
        return sine_waves, uv


class DeterministicSourceModuleHnNSF(nn.Module):
    def __init__(
        self,
        sampling_rate: int,
        harmonic_num: int = 0,
        sine_amp: float = 0.1,
        add_noise_std: float = 0.003,
        voiced_threshold: float = 0,
    ):
        super().__init__()
        self.l_sin_gen = DeterministicSineGen(
            sampling_rate,
            harmonic_num,
            sine_amp,
            add_noise_std,
            voiced_threshold,
        )
        self.l_linear = nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = nn.Tanh()

    def forward(
        self,
        f0: torch.Tensor,
        phase: torch.Tensor,
        noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sine_wavs, uv = self.l_sin_gen(f0, phase, noise)
        sine_wavs = sine_wavs.to(
            device=self.l_linear.weight.device, dtype=self.l_linear.weight.dtype
        )
        source = self.l_linear(sine_wavs.transpose(1, 2))
        source = self.l_tanh(source).transpose(1, 2)
        return source, uv
