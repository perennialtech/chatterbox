import pytest
import torch

import chatterbox.vc.backends.torch_backend as module
from chatterbox.models.s3gen.conditioning import ConditioningError
from chatterbox.vc.errors import VoiceConditioningError
from chatterbox.vc.service import ChatterboxVC


class FakeS3Gen:
    def __init__(self):
        self.calls = 0

    def embed_ref(self, wav, sr, device):
        self.calls += 1
        return object()

    def prepare_ref_condition(self, condition):
        return condition


@pytest.fixture
def setup_voice(monkeypatch, tmp_path):
    monkeypatch.setattr(module, "load_wav_24k", lambda *a, **kw: torch.zeros(24))
    target = tmp_path / "target.wav"
    target.write_bytes(b"reference")
    return module.TorchVCBackend(FakeS3Gen(), "cpu"), target


def test_reuses_only_the_active_unchanged_reference(setup_voice):
    backend, target = setup_voice
    backend.set_target_voice_path(str(target))
    original = backend.ref_condition
    backend.set_target_voice_path(target)
    assert backend.ref_condition is original
    assert backend.s3gen.calls == 1

    other = target.with_name("other.wav")
    other.write_bytes(b"other")
    backend.set_target_voice_path(other)
    backend.set_target_voice_path(target)
    assert backend.s3gen.calls == 3


def test_file_changes_and_replacement_invalidate_cache(setup_voice):
    backend, target = setup_voice
    backend.set_target_voice_path(target)
    target.write_bytes(b"updated reference")
    backend.set_target_voice_path(target)
    replacement = target.with_name("replacement.wav")
    replacement.write_bytes(target.read_bytes())
    replacement.replace(target)
    backend.set_target_voice_path(target)
    assert backend.s3gen.calls == 3


def test_symlink_retargeting_invalidates_cache(setup_voice):
    backend, target = setup_voice
    link = target.with_name("link.wav")
    link.symlink_to(target)
    backend.set_target_voice_path(link)
    backend.set_target_voice_path(target)
    assert backend.s3gen.calls == 1
    other = target.with_name("other.wav")
    other.write_bytes(b"other")
    link.unlink()
    link.symlink_to(other)
    backend.set_target_voice_path(link)
    assert backend.s3gen.calls == 2


def test_missing_reference_does_not_reuse_stale_condition(setup_voice):
    backend, target = setup_voice
    backend.set_target_voice_path(target)
    original = backend.ref_condition
    target.unlink()
    with pytest.raises(FileNotFoundError):
        backend.set_target_voice_path(target)
    assert backend.ref_condition is original


def test_explicit_condition_clears_path_cache(setup_voice):
    backend, target = setup_voice
    backend.set_target_voice_path(target)
    backend.set_target_voice_condition(object())
    backend.set_target_voice_path(target)
    assert backend.s3gen.calls == 2


def test_invalid_reference_preserves_active_voice(setup_voice, monkeypatch):
    backend, target = setup_voice
    backend.set_target_voice_path(target)
    original = backend.ref_condition
    target.write_bytes(b"bad reference")

    def fail(*a, **kw):
        raise ConditioningError("invalid voice")

    monkeypatch.setattr(backend.s3gen, "embed_ref", fail)
    with pytest.raises(VoiceConditioningError, match="invalid voice"):
        backend.set_target_voice_path(target)
    assert backend.ref_condition is original


def test_changes_during_embedding_are_not_cached(setup_voice, monkeypatch):
    backend, target = setup_voice
    embed = backend.s3gen.embed_ref

    def mutate(*a, **kw):
        target.write_bytes(target.read_bytes() + b"!")
        return embed(*a, **kw)

    monkeypatch.setattr(backend.s3gen, "embed_ref", mutate)
    backend.set_target_voice_path(target)
    backend.set_target_voice_path(target)
    assert backend.s3gen.calls == 2


def test_tensor_generation_honors_target_path(setup_voice, monkeypatch):
    backend, target = setup_voice
    result = module.VCResult(torch.ones(1, 8), 24_000, {})
    monkeypatch.setattr(backend, "convert_from_tensors", lambda *a, **kw: result)
    model = ChatterboxVC(backend)
    wav, sr, _ = model.generate(torch.zeros(16), target_voice_path=target)
    assert backend.s3gen.calls == 1
    assert wav is result.wav
    assert sr == 24_000


def test_warmup_uses_the_active_target(setup_voice, monkeypatch):
    backend, target = setup_voice
    calls = []
    monkeypatch.setattr(
        backend.s3gen, "warmup", lambda **kw: calls.append(kw), raising=False
    )
    model = ChatterboxVC(backend)
    with pytest.raises(VoiceConditioningError, match="Target voice is not set"):
        model.warmup()
    model.set_target_voice_path(target)
    model.warmup()
    assert calls == [{"ref_condition": backend.ref_condition}]
