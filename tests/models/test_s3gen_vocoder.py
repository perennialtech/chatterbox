import torch

from chatterbox.models.s3gen.stft import RealISTFT, RealSTFT
from chatterbox.models.s3gen.vocoder import HiFTGenerator


class ConstantF0Predictor(torch.nn.Module):
    def forward(self, x):
        return torch.full(
            (x.size(0), x.size(-1)), 120.0, device=x.device, dtype=x.dtype
        )


def test_hift_inference_output_length_matches_mel_count_times_source_hop():
    vocoder = HiFTGenerator(
        base_channels=16,
        nb_harmonics=2,
        sampling_rate=24000,
        upsample_rates=[2],
        upsample_kernel_sizes=[4],
        istft_params={"n_fft": 8, "hop_len": 2},
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1]],
        source_resblock_kernel_sizes=[7],
        source_resblock_dilation_sizes=[[1]],
        f0_predictor=ConstantF0Predictor(),
    ).eval()

    num_mels = 5
    speech_feat = torch.randn(1, 80, num_mels)
    phase = torch.zeros(1, vocoder.nb_harmonics + 1, 1)
    noise = torch.zeros(1, vocoder.nb_harmonics + 1, num_mels * vocoder.source_hop)

    wav, source = vocoder.inference(
        speech_feat,
        source_phase=phase,
        source_noise=noise,
    )

    assert wav.shape == (1, num_mels * vocoder.source_hop)
    assert source.shape == (1, 1, num_mels * vocoder.source_hop)


def _roundtrip(x):
    stft = RealSTFT(n_fft=16, hop_len=4, center=True)
    istft = RealISTFT(n_fft=16, hop_len=4, center=True)

    real, imag = stft(x)
    magnitude = torch.sqrt(real.square() + imag.square())
    phase = torch.atan2(imag, real)
    return istft(magnitude, phase)


def test_real_stft_istft_reconstructs_sinusoid():
    t = torch.arange(0, 128, dtype=torch.float32)
    x = torch.sin(2 * torch.pi * t / 16).unsqueeze(0)

    y = _roundtrip(x)

    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-4, rtol=1e-4)


def test_real_stft_istft_reconstructs_random_waveform():
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(1, 128, generator=generator)

    y = _roundtrip(x)

    assert y.shape == x.shape
    assert torch.allclose(y, x, atol=1e-4, rtol=1e-4)
