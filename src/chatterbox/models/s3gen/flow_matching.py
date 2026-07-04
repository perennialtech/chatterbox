from abc import ABC

import torch
import torch.nn.functional as F

from .configs import CFM_PARAMS


def cast_all(*args, dtype):
    return [
        (
            a
            if a is None or (not a.dtype.is_floating_point) or a.dtype == dtype
            else a.to(dtype)
        )
        for a in args
    ]


class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        n_feats,
        cfm_params,
        n_spks=1,
        spk_emb_dim=128,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        if hasattr(cfm_params, "sigma_min"):
            self.sigma_min = cfm_params.sigma_min
        else:
            self.sigma_min = 1e-4
        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None):
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(
            z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond
        )

    def solve_euler(self, x, t_span, mu, mask, spks, cond):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1]

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        b, _, t = mu.shape
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z
        loss = F.mse_loss(
            self.estimator(y, mask, mu, t.squeeze(), spks), u, reduction="sum"
        ) / (torch.sum(mask) * u.shape[1])
        return loss, y


class ConditionalCFM(BASECFM):
    def __init__(
        self,
        in_channels,
        cfm_params,
        n_spks=1,
        spk_emb_dim=64,
        estimator: torch.nn.Module = None,
    ):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
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


class CausalConditionalCFM(ConditionalCFM):
    def __init__(
        self,
        in_channels=240,
        cfm_params=CFM_PARAMS,
        n_spks=1,
        spk_emb_dim=80,
        estimator=None,
    ):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        self.rand_noise = None
