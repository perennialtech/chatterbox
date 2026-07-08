import torch

from chatterbox.models.s3gen.pipeline import S3Token2Mel, S3Token2Wav


class DummyFlow(torch.nn.Module):
    token_mel_ratio = 2
    pre_lookahead_len = 3

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.last_token = None
        self.last_token_len = None

    def inference(self, token, token_len, **kwargs):
        self.last_token = token.detach().clone()
        self.last_token_len = token_len.detach().clone()
        mel_len = int(token_len.max().item()) * self.token_mel_ratio
        return torch.zeros(token.size(0), 80, mel_len, device=token.device), None


class DummyToken2Mel(S3Token2Mel):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.flow = DummyFlow()
        self.meanflow = False

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def prepare_ref_dict(self, ref_dict):
        return ref_dict


class DummyVocoder(torch.nn.Module):
    source_hop = 4

    def __init__(self):
        super().__init__()
        self.last_mel_len = None

    def inference(self, speech_feat, cache_source=None):
        self.last_mel_len = speech_feat.size(-1)
        samples = speech_feat.size(-1) * self.source_hop
        return (
            torch.ones(speech_feat.size(0), samples, device=speech_feat.device),
            torch.ones(speech_feat.size(0), 1, samples, device=speech_feat.device),
        )


class DummyToken2Wav(S3Token2Wav):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.flow = DummyFlow()
        self.mel2wav = DummyVocoder()
        self.meanflow = False
        self.register_buffer("trim_fade", torch.ones(1), persistent=False)
        self.register_buffer("end_fade", torch.ones(1), persistent=False)

    @property
    def device(self):
        return torch.device("cpu")

    @property
    def dtype(self):
        return torch.float32

    def prepare_ref_dict(self, ref_dict):
        return ref_dict


def _ref_dict():
    return {
        "prompt_token": torch.zeros(1, 1, dtype=torch.long),
        "prompt_token_len": torch.tensor([1], dtype=torch.long),
        "prompt_feat": torch.zeros(1, 2, 80),
        "embedding": torch.zeros(1, 192),
    }


def test_s3token2mel_forward_accepts_1d_token_vector_as_batch_one():
    model = DummyToken2Mel()
    output = model(
        torch.tensor([1, 2, 3]),
        ref_wav=None,
        ref_sr=None,
        ref_dict=_ref_dict(),
    )

    assert model.flow.last_token.shape == (1, 3)
    assert model.flow.last_token_len.tolist() == [3]
    assert output.shape == (1, 80, 6)


def test_s3token2wav_inference_keeps_final_context_for_vocoder_then_trims_samples():
    model = DummyToken2Wav()
    wav, source = model.inference(
        torch.tensor([1, 2, 3]),
        ref_dict=_ref_dict(),
        drop_invalid_tokens=False,
    )

    original_mel_len = 3 * model._token_mel_ratio
    context_mel_len = model._final_context_token_count * model._token_mel_ratio

    assert model.flow.last_token_len.tolist() == [6]
    assert model.mel2wav.last_mel_len == original_mel_len + context_mel_len
    assert wav.shape == (1, original_mel_len * model.mel2wav.source_hop)
    assert source.shape == (1, 1, original_mel_len * model.mel2wav.source_hop)
