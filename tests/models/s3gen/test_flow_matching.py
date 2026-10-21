import pytest
import torch

from chatterbox.models.s3gen.flow_matching import ConditionalCFM


class ZeroEstimator(torch.nn.Module):
    dtype = torch.float32

    def forward(self, x, *args):
        return torch.zeros_like(x)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_solver_uses_the_same_fp32_noise_under_autocast(dtype):
    decoder = ConditionalCFM(80, estimator=ZeroEstimator())
    mu = torch.zeros(1, 80, 8, dtype=dtype)
    torch.manual_seed(1234)
    expected_noise = torch.randn_like(mu, dtype=torch.float32)
    torch.manual_seed(1234)
    result, _ = decoder(mu, torch.ones(1, 1, 8), 2)
    assert result.dtype == torch.float32
    assert torch.equal(result, expected_noise)
