from __future__ import annotations

import torch


class AudioPostprocessor(torch.nn.Module):
    def __init__(self, sample_rate: int = 24_000, fade_ms: int = 20):
        super().__init__()
        n = sample_rate * fade_ms // 1000
        fade = torch.zeros(2 * n)
        fade[n:] = (torch.cos(torch.linspace(torch.pi, 0, n)) + 1.0) / 2.0
        self.register_buffer("fade", fade, persistent=False)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.size(-1) >= self.fade.numel():
            wav = wav.clone()
            wav[..., : self.fade.numel()] *= self.fade.to(
                device=wav.device, dtype=wav.dtype
            )
        return wav
