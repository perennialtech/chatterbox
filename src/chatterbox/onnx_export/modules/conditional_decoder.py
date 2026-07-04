import torch

from ..dynamic_axes import CONDITIONAL_DECODER_DYNAMIC_AXES

input_names = ["x", "mask", "mu", "spks", "cond", "t", "r"]
output_names = ["dxdt"]
dynamic_axes = CONDITIONAL_DECODER_DYNAMIC_AXES


class ConditionalDecoderStepExport(torch.nn.Module):
    def __init__(self, estimator: torch.nn.Module):
        super().__init__()
        self.estimator = estimator

    def forward(self, x, mask, mu, spks, cond, t, r):
        return self.estimator.forward_export(
            x=x,
            mask=mask,
            mu=mu,
            spks=spks,
            cond=cond,
            t=t,
            r=r,
        )


def make_dummy_inputs(batch: int = 1, mel_frames: int = 64, dtype=torch.float32):
    return (
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.ones(batch, 1, mel_frames, dtype=dtype),
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.randn(batch, 80, dtype=dtype),
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.zeros(batch, dtype=dtype),
        torch.full((batch,), 0.5, dtype=dtype),
    )
