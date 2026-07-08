import torch


def cast_all(*args, dtype):
    return [
        (
            a
            if a is None or (not a.dtype.is_floating_point) or a.dtype == dtype
            else a.to(dtype)
        )
        for a in args
    ]


class ConditionalCFM(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        estimator: torch.nn.Module = None,
    ):
        super().__init__()
        self.estimator = estimator

    def make_t_span(
        self,
        n_timesteps: int,
        device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)
        return t_span

    @torch.inference_mode()
    def forward(
        self,
        mu,
        mask,
        n_timesteps,
        temperature=1.0,
        spks=None,
        cond=None,
    ):
        noise = torch.randn_like(mu) * temperature
        t_span = self.make_t_span(n_timesteps, mu.device, mu.dtype)
        return (
            self.decode_from_noise(
                x=noise,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
                t_span=t_span,
            ),
            None,
        )

    def decode_from_noise(self, x, mu, mask, spks, cond, t_span):
        in_dtype = x.dtype
        x, t_span, mu, mask, spks, cond = cast_all(
            x, t_span, mu, mask, spks, cond, dtype=self.estimator.dtype
        )

        for t, r in zip(t_span[:-1], t_span[1:]):
            t_batch = t.expand(x.size(0))
            r_batch = r.expand(x.size(0))
            dxdt = self.estimator(x, mask, mu, t_batch, spks, cond, r_batch)
            x = x + (r - t) * dxdt

        return x.to(in_dtype)
