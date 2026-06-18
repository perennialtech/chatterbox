from __future__ import annotations

import torch


class MeanFlowSampler(torch.nn.Module):
    def __init__(self, estimator: torch.nn.Module, steps: int = 2):
        super().__init__()
        self.estimator = estimator
        self.steps = steps

    @torch.inference_mode()
    def sample(
        self,
        *,
        mu: torch.Tensor,
        mask: torch.Tensor,
        speaker: torch.Tensor,
        cond: torch.Tensor,
        steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        n_steps = steps or self.steps
        x = torch.randn(
            mu.shape,
            device=mu.device,
            dtype=mu.dtype,
            generator=generator,
        )

        t_span = torch.linspace(0.0, 1.0, n_steps + 1, device=mu.device, dtype=mu.dtype)
        for t, r in zip(t_span[:-1], t_span[1:]):
            t_batch = t.expand(mu.size(0))
            r_batch = r.expand(mu.size(0))
            velocity = self.estimator(
                x=x,
                mask=mask,
                mu=mu,
                t=t_batch,
                spks=speaker,
                cond=cond,
                r=r_batch,
            )
            x = x + (r - t) * velocity

        return x
