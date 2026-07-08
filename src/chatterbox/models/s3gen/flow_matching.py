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
        cfm_params,
        estimator: torch.nn.Module = None,
    ):
        super().__init__()
        self.t_scheduler = cfm_params.t_scheduler
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.estimator = estimator

    def make_t_span(
        self,
        n_timesteps: int,
        device,
        dtype: torch.dtype,
        meanflow: bool = False,
    ) -> torch.Tensor:
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=device, dtype=dtype)
        if (not meanflow) and self.t_scheduler == "cosine":
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
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
        meanflow=False,
    ):
        noise = torch.randn_like(mu) * temperature
        t_span = self.make_t_span(n_timesteps, mu.device, mu.dtype, meanflow=meanflow)
        return (
            self.decode_from_noise(
                noise=noise,
                mu=mu,
                mask=mask,
                spks=spks,
                cond=cond,
                t_span=t_span,
                meanflow=meanflow,
            ),
            None,
        )

    def decode_from_noise(self, noise, mu, mask, spks, cond, t_span, meanflow=False):
        if meanflow:
            return self.basic_euler(noise, t_span, mu, mask, spks, cond)
        return self.solve_euler(noise, t_span, mu, mask, spks, cond, meanflow=meanflow)

    def solve_euler(self, x, t_span, mu, mask, spks, cond, meanflow=False):
        in_dtype = x.dtype
        x, t_span, mu, mask, spks, cond = cast_all(
            x, t_span, mu, mask, spks, cond, dtype=self.estimator.dtype
        )

        batch, feat_dim, timesteps = x.size()
        spk_dim = spks.size(1)
        cond_dim = cond.size(1)

        x_in = x.new_zeros([2 * batch, feat_dim, timesteps])
        mask_in = mask.new_zeros([2 * batch, mask.size(1), timesteps])
        mu_in = mu.new_zeros([2 * batch, mu.size(1), timesteps])
        t_in = x.new_zeros([2 * batch])
        spks_in = spks.new_zeros([2 * batch, spk_dim])
        cond_in = cond.new_zeros([2 * batch, cond_dim, timesteps])
        r_in = x.new_zeros([2 * batch])

        for t, r in zip(t_span[:-1], t_span[1:]):
            t = t.unsqueeze(0)
            r = r.unsqueeze(0)

            x_in[:batch] = x_in[batch:] = x
            mask_in[:batch] = mask_in[batch:] = mask
            # The second half is the unconditional CFG branch: mu, speaker,
            # and acoustic conditioning are intentionally left zeroed.
            mu_in[:batch] = mu
            t_in[:batch] = t_in[batch:] = t
            spks_in[:batch] = spks
            cond_in[:batch] = cond
            r_in[:batch] = r_in[batch:] = r

            dxdt = self.estimator(
                x_in, mask_in, mu_in, t_in, spks_in, cond_in, r_in if meanflow else None
            )

            dxdt, cfg_dxdt = torch.split(dxdt, [batch, batch], dim=0)
            dxdt = (
                1.0 + self.inference_cfg_rate
            ) * dxdt - self.inference_cfg_rate * cfg_dxdt
            x = x + (r - t) * dxdt

        return x.to(in_dtype)

    def basic_euler(self, x, t_span, mu, mask, spks, cond):
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
