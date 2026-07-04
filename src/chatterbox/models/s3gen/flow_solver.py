import torch
import torch.nn as nn


class MeanflowSolver(nn.Module):
    def __init__(self, decoder: nn.Module, meanflow: bool = True):
        super().__init__()
        self.decoder = decoder
        self.meanflow = meanflow

    def make_t_span(self, n_timesteps: int, device, dtype) -> torch.Tensor:
        return torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)

    def forward_from_noise(
        self,
        noise: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        t_span: torch.Tensor,
    ) -> torch.Tensor:
        x = noise
        for t, r in zip(t_span[:-1], t_span[1:]):
            t_batch = t.expand(x.size(0))
            r_batch = r.expand(x.size(0))
            dxdt = self.decoder.estimator.forward_export(
                x=x,
                mask=mask,
                mu=mu,
                spks=spks,
                cond=cond,
                t=t_batch,
                r=r_batch,
            )
            x = x + (r - t) * dxdt
        return x
