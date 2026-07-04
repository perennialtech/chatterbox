import torch

from ..dynamic_axes import FLOW_DECODER_DYNAMIC_AXES

input_names = ["noise", "mask", "mu", "spks", "cond"]
output_names = ["mel"]
dynamic_axes = FLOW_DECODER_DYNAMIC_AXES


class FlowDecoderMeanflow2Export(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.decoder = decoder
        self.register_buffer(
            "t_span", torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)
        )

    def forward(self, noise, mask, mu, spks, cond):
        x = noise
        for i in range(2):
            t = self.t_span[i].to(dtype=x.dtype).expand(x.size(0))
            r = self.t_span[i + 1].to(dtype=x.dtype).expand(x.size(0))
            dxdt = self.decoder.estimator.forward_export(
                x=x,
                mask=mask,
                mu=mu,
                spks=spks,
                cond=cond,
                t=t,
                r=r,
            )
            x = x + (r[:1] - t[:1]) * dxdt
        return x


def make_dummy_inputs(batch: int = 1, mel_frames: int = 64, dtype=torch.float32):
    return (
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.ones(batch, 1, mel_frames, dtype=dtype),
        torch.randn(batch, 80, mel_frames, dtype=dtype),
        torch.randn(batch, 80, dtype=dtype),
        torch.randn(batch, 80, mel_frames, dtype=dtype),
    )
