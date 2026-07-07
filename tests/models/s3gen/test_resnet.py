import torch
import torch.nn as nn

from chatterbox.models.s3gen.resnet import MaskedGroupNorm1D


def test_masked_group_norm_matches_group_norm_with_all_ones_mask():
    torch.manual_seed(0)

    x = torch.randn(3, 4, 5)
    mask = torch.ones(3, 1, 5)

    norm = MaskedGroupNorm1D(num_groups=2, num_channels=4)
    expected_norm = nn.GroupNorm(num_groups=2, num_channels=4)

    with torch.no_grad():
        norm.weight.copy_(torch.randn(4))
        norm.bias.copy_(torch.randn(4))
        expected_norm.weight.copy_(norm.weight)
        expected_norm.bias.copy_(norm.bias)

    output = norm(x, mask)
    expected = expected_norm(x)

    torch.testing.assert_close(output, expected)


def test_masked_group_norm_ignores_masked_timesteps():
    torch.manual_seed(0)

    x = torch.randn(2, 4, 5)
    mask = torch.tensor(
        [
            [[1.0, 1.0, 1.0, 0.0, 0.0]],
            [[1.0, 1.0, 0.0, 0.0, 0.0]],
        ]
    )

    norm = MaskedGroupNorm1D(num_groups=2, num_channels=4)

    x_changed = x + (1.0 - mask) * 1000.0

    output = norm(x, mask)
    changed_output = norm(x_changed, mask)

    torch.testing.assert_close(changed_output, output)


def test_masked_group_norm_all_zero_mask_returns_finite_zeros():
    torch.manual_seed(0)

    x = torch.randn(2, 4, 5)
    mask = torch.zeros(2, 1, 5)

    norm = MaskedGroupNorm1D(num_groups=2, num_channels=4)
    output = norm(x, mask)

    assert torch.isfinite(output).all()
    torch.testing.assert_close(output, torch.zeros_like(output))
