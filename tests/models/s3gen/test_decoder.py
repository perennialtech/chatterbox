import pytest
import torch

from chatterbox.models.s3gen.decoder import CausalConv1d, ConditionalDecoder


def test_dilated_causal_conv_preserves_length_and_is_future_independent():
    conv = CausalConv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        dilation=2,
        bias=False,
    )

    with torch.no_grad():
        conv.weight.copy_(torch.tensor([[[1.0, 2.0, 3.0]]]))

    x = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    y = conv(x)

    assert y.shape == x.shape

    cutoff = 4
    x_changed = x.clone()
    x_changed[:, :, cutoff + 1 :] += 1000.0
    y_changed = conv(x_changed)

    torch.testing.assert_close(
        y_changed[:, :, : cutoff + 1],
        y[:, :, : cutoff + 1],
    )


def test_conditional_decoder_accepts_cpu_scalar_time_conditions_on_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")

    device = torch.device("cuda")
    batch_size = 1
    time_steps = 4

    model = ConditionalDecoder(
        in_channels=8,
        out_channels=2,
        channels=[8],
        n_blocks=0,
        num_mid_blocks=1,
        num_heads=1,
        attention_head_dim=8,
    ).to(device)
    model.eval()

    x = torch.randn(batch_size, 2, time_steps, device=device)
    mu = torch.randn(batch_size, 2, time_steps, device=device)
    spks = torch.randn(batch_size, 2, device=device)
    cond = torch.randn(batch_size, 2, time_steps, device=device)
    mask = torch.ones(batch_size, 1, time_steps, device=device)

    t = torch.tensor(0.25)
    r = torch.tensor(0.75)

    with torch.inference_mode():
        output = model(x=x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)

    assert output.device.type == device.type
    assert output.shape == (batch_size, 2, time_steps)
    assert torch.isfinite(output).all()


def test_conditional_decoder_rebuilds_attention_masks_across_resolutions_on_cpu():
    torch.manual_seed(0)

    batch_size = 2
    time_steps = 5

    model = ConditionalDecoder(
        in_channels=8,
        out_channels=2,
        channels=[8, 16],
        n_blocks=1,
        num_mid_blocks=1,
        num_heads=1,
        attention_head_dim=8,
    )
    model.eval()

    x = torch.randn(batch_size, 2, time_steps)
    mu = torch.randn(batch_size, 2, time_steps)
    spks = torch.randn(batch_size, 2)
    cond = torch.randn(batch_size, 2, time_steps)
    mask = torch.tensor(
        [
            [[1.0, 1.0, 1.0, 1.0, 1.0]],
            [[1.0, 1.0, 1.0, 0.0, 0.0]],
        ]
    )

    t = torch.tensor([0.25, 0.75])
    r = torch.tensor([0.0])

    with torch.inference_mode():
        output = model(x=x, mask=mask, mu=mu, t=t, spks=spks, cond=cond, r=r)

    assert output.shape == (batch_size, 2, time_steps)
    assert torch.isfinite(output).all()
