import sys
import types

import pytest
import torch

import chatterbox.models.s3gen.pipeline as pipeline
import chatterbox.vc.backends.torch_backend as torch_backend
from chatterbox.audio import S3_SR, S3_TOKEN_RATE
from chatterbox.models.s3gen.conditioning import (ConditioningError,
                                                  S3ReferenceCondition)
from chatterbox.models.s3gen.pipeline import (FLOW_CHUNK_TOKENS,
                                              REF_MAX_PROMPT_TOKENS,
                                              S3Token2Mel, S3Token2Wav)
from chatterbox.vc.errors import VoiceConditioningError


class DummyFlow(torch.nn.Module):
    token_mel_ratio = 2
    pre_lookahead_len = 3

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def inference(self, token, token_len, **kwargs):
        self.calls.append(
            {
                "token": token.detach().clone(),
                "token_len": token_len.detach().clone(),
            }
        )
        mel_len = int(token_len.max().item()) * self.token_mel_ratio
        return torch.zeros(token.size(0), 80, mel_len, device=token.device), None

    @property
    def last_token(self):
        return self.calls[-1]["token"]

    @property
    def last_token_len(self):
        return self.calls[-1]["token_len"]


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


class DummyVocoder(torch.nn.Module):
    source_hop = 4

    def __init__(self):
        super().__init__()
        self.last_mel_len = None
        self.call_count = 0

    def inference(self, speech_feat, cache_source=None):
        self.call_count += 1
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


class FakeSpeakerEncoder(torch.nn.Module):
    def forward(self, fbank):
        return torch.zeros(fbank.size(0), 192, device=fbank.device, dtype=fbank.dtype)


class FakeTokenizer(torch.nn.Module):
    def forward(self, wav):
        hop = int(S3_SR // S3_TOKEN_RATE)
        token_len = wav.size(1) // hop
        return (
            torch.zeros(wav.size(0), token_len, dtype=torch.long, device=wav.device),
            torch.full((wav.size(0),), token_len, dtype=torch.long, device=wav.device),
        )


def _ref_dict(prompt_tokens: int = 1):
    return {
        "prompt_token": torch.zeros(1, prompt_tokens, dtype=torch.long),
        "prompt_token_len": torch.tensor([prompt_tokens], dtype=torch.long),
        "prompt_feat": torch.zeros(1, prompt_tokens * 2, 80),
        "embedding": torch.zeros(1, 192),
    }


def test_s3token2mel_inference_accepts_1d_token_vector_as_batch_one():
    model = DummyToken2Mel()
    output = model.inference(
        torch.tensor([1, 2, 3]),
        ref_wav=None,
        ref_sr=None,
        ref_dict=_ref_dict(),
        drop_invalid_tokens=False,
    )

    assert model.flow.last_token.shape == (1, 6)
    assert model.flow.last_token_len.tolist() == [6]
    assert output.shape == (1, 80, 6)


@pytest.mark.parametrize("num_tokens", [1, 2, 3])
def test_s3token2wav_inference_short_targets_have_exact_trimmed_sample_count(
    num_tokens,
):
    model = DummyToken2Wav()
    wav, source = model.inference(
        torch.arange(num_tokens),
        ref_dict=_ref_dict(),
        drop_invalid_tokens=False,
    )

    original_mel_len = num_tokens * model._token_mel_ratio
    expected_samples = original_mel_len * model.mel2wav.source_hop

    assert wav.shape == (1, expected_samples)
    assert source.shape == (1, 1, expected_samples)
    assert model.mel2wav.last_mel_len == original_mel_len
    assert model.mel2wav.call_count == 1


def test_s3token2wav_forward_generates_full_utterance_without_lookahead_truncation():
    model = DummyToken2Wav()
    wav, source = model(
        torch.tensor([1, 2, 3]),
        ref_dict=_ref_dict(),
        drop_invalid_tokens=False,
    )

    original_mel_len = 3 * model._token_mel_ratio
    expected_samples = original_mel_len * model.mel2wav.source_hop

    assert wav.shape == (1, expected_samples)
    assert source.shape == (1, 1, expected_samples)
    assert model.mel2wav.last_mel_len == original_mel_len


def test_s3token2wav_rejects_empty_tokens_after_invalid_token_drop():
    model = DummyToken2Wav()

    with pytest.raises(ValueError, match="At least one valid speech token"):
        model.inference(
            torch.tensor([-1, 999999]),
            ref_dict=_ref_dict(),
            drop_invalid_tokens=True,
        )


def test_direct_ref_dict_prompt_is_capped_like_reference_audio():
    model = DummyToken2Wav()
    condition = model.prepare_ref_condition(_ref_dict(REF_MAX_PROMPT_TOKENS + 25))

    assert condition.prompt_token.shape == (1, REF_MAX_PROMPT_TOKENS)
    assert condition.prompt_token_len.tolist() == [REF_MAX_PROMPT_TOKENS]
    assert condition.prompt_feat.shape == (1, REF_MAX_PROMPT_TOKENS * 2, 80)


def test_long_ref_wav_is_center_cropped_and_prompt_capped(monkeypatch):
    model = DummyToken2Mel()
    model.tokenizer = FakeTokenizer()
    model.speaker_encoder = FakeSpeakerEncoder()
    model.mel_extractor = lambda wav: torch.zeros(
        wav.size(0),
        80,
        wav.size(1) // int(S3_SR // (S3_TOKEN_RATE * 2)),
        device=wav.device,
        dtype=wav.dtype,
    )

    monkeypatch.setattr(
        pipeline, "resample_audio", lambda wav, src_sr, dst_sr, device: wav
    )
    monkeypatch.setattr(
        pipeline,
        "extract_fbank_features",
        lambda wav: torch.zeros(
            wav.size(0), 10, 80, device=wav.device, dtype=wav.dtype
        ),
    )

    ref_wav = torch.zeros(15 * S3_SR)
    condition = model.embed_ref(ref_wav, S3_SR)

    assert condition.prompt_token.size(1) <= REF_MAX_PROMPT_TOKENS
    assert condition.prompt_token_len.tolist() == [REF_MAX_PROMPT_TOKENS]
    assert condition.prompt_feat.size(1) == REF_MAX_PROMPT_TOKENS * 2


@pytest.mark.parametrize("seconds", [5, 15, 30])
def test_chunked_flow_returns_the_exact_target_mel_length(seconds):
    model = DummyToken2Wav()
    token_count = seconds * int(S3_TOKEN_RATE)
    tokens = torch.arange(token_count)

    output, original_mel_len = model._chunked_flow_inference_impl(
        tokens,
        ref_dict=_ref_dict(),
    )

    expected_chunks = (token_count + FLOW_CHUNK_TOKENS - 1) // FLOW_CHUNK_TOKENS
    expected_mel_len = token_count * model._token_mel_ratio

    assert output.shape == (1, 80, expected_mel_len)
    assert original_mel_len == expected_mel_len
    assert len(model.flow.calls) == expected_chunks


def test_flow_generation_is_always_chunked_for_long_inputs():
    model = DummyToken2Wav()
    token_count = FLOW_CHUNK_TOKENS * 2 + 1

    output = model.flow_inference(
        torch.arange(token_count),
        ref_dict=_ref_dict(),
    )

    assert output.shape == (1, 80, token_count * model._token_mel_ratio)
    assert len(model.flow.calls) == 3


def test_apply_output_fades_uses_non_overlapping_short_fades():
    model = DummyToken2Wav()
    model.trim_fade = torch.cat([torch.zeros(8), torch.ones(8)])
    model.end_fade = torch.linspace(1.0, 0.0, 16)

    wav = torch.ones(1, 8)
    faded = model._apply_output_fades(wav.clone())

    assert torch.all(faded > 0.0)
    assert torch.isfinite(faded).all()


class FakeBackendS3Gen:
    dtype = torch.float32

    def __init__(self):
        self.prepared_conditions = []
        self.inference_calls = []
        self.embedded_references = []

    def prepare_ref_condition(self, condition):
        if isinstance(condition, S3ReferenceCondition):
            condition.validate()
        else:
            S3ReferenceCondition.from_mapping(
                condition,
                device="cpu",
                dtype=torch.float32,
            )

        prepared = {"prepared": condition}
        self.prepared_conditions.append(prepared)
        return prepared

    def tokenizer(self, audio):
        return (
            torch.tensor([[1, 2]], dtype=torch.long, device=audio.device),
            torch.tensor([2], dtype=torch.long, device=audio.device),
        )

    def inference(self, **kwargs):
        self.inference_calls.append(kwargs)
        return (
            torch.ones(1, 8, device=kwargs["speech_tokens"].device),
            torch.zeros(1, 1, 8, device=kwargs["speech_tokens"].device),
        )

    def embed_ref(self, wav, sample_rate, device):
        self.embedded_references.append((wav, sample_rate, device))
        return {"embedded": True}


def _voice_condition(prompt_width=3, prompt_len=3, prompt_feat_width=None):
    if prompt_feat_width is None:
        prompt_feat_width = prompt_width * 2
    return {
        "prompt_token": torch.zeros(1, prompt_width, dtype=torch.long),
        "prompt_token_len": torch.tensor([prompt_len], dtype=torch.long),
        "prompt_feat": torch.zeros(1, prompt_feat_width, 80),
        "embedding": torch.zeros(1, 192),
    }


def test_torch_backend_rejects_invalid_target_condition_without_replacing_active_target():
    model = FakeBackendS3Gen()
    backend = torch_backend.TorchVCBackend(model, "cpu")

    backend.set_target_voice_condition(_voice_condition())
    active_target = backend.ref_condition

    with pytest.raises(VoiceConditioningError) as error:
        backend.set_target_voice_condition(
            _voice_condition(
                prompt_width=4,
                prompt_len=3,
                prompt_feat_width=6,
            )
        )

    assert isinstance(error.value.__cause__, ConditioningError)
    assert backend.ref_condition is active_target


def test_torch_backend_uses_s3gen_inference_with_token_lengths_and_prepared_target():
    model = FakeBackendS3Gen()
    backend = torch_backend.TorchVCBackend(model, "cpu")
    backend.set_target_voice_condition(_voice_condition())
    active_target = backend.ref_condition

    result = backend.convert_from_tensors(torch.zeros(1, 16))

    assert torch.equal(result.wav, torch.ones(1, 8))
    assert result.sample_rate == 24_000
    assert len(model.inference_calls) == 1
    call = model.inference_calls[0]
    assert call["speech_token_lens"].tolist() == [2]
    assert call["ref_dict"] is active_target
    assert call["drop_invalid_tokens"] is False


def test_torch_backend_path_reference_is_not_pretruncated(monkeypatch):
    model = FakeBackendS3Gen()
    backend = torch_backend.TorchVCBackend(model, "cpu")
    requested_max_lengths = []

    def fake_load_wav_24k(path, device, max_len=None):
        requested_max_lengths.append(max_len)
        return torch.zeros(1, 32)

    monkeypatch.setattr(torch_backend, "load_wav_24k", fake_load_wav_24k)
    monkeypatch.setattr(
        torch_backend,
        "load_wav_16k",
        lambda path, device: torch.zeros(1, 16),
    )

    backend.convert_from_path("source.wav", target_voice_path="target.wav")

    assert requested_max_lengths == [None]
    assert len(model.embedded_references) == 1


def test_download_pretrained_checkpoint_accepts_snapshot_without_builtin_condition(
    monkeypatch,
    tmp_path,
):
    (tmp_path / "s3gen_meanflow.safetensors").touch()
    calls = []

    def snapshot_download(**kwargs):
        calls.append(kwargs)
        return str(tmp_path)

    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    checkpoint_dir = torch_backend.download_pretrained_checkpoint("example/repo")

    assert checkpoint_dir == tmp_path
    assert calls == [
        {
            "repo_id": "example/repo",
            "allow_patterns": ["s3gen_meanflow.safetensors", "conds.pt"],
        }
    ]


def test_download_pretrained_checkpoint_requires_model_file(monkeypatch, tmp_path):
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = lambda **kwargs: str(tmp_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)

    with pytest.raises(FileNotFoundError, match="s3gen_meanflow.safetensors"):
        torch_backend.download_pretrained_checkpoint("example/repo")
