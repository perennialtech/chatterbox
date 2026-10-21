from types import SimpleNamespace

import pytest
import torch

from scripts.benchmark_latency import configure_device, percentile


@pytest.mark.parametrize("name,index", [("cuda", 2), ("cuda:1", 1)])
def test_memory_cap_receives_an_explicit_device_index(monkeypatch, name, index):
    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(total_memory=8 * 2**30),
    )
    monkeypatch.setattr(
        torch.cuda,
        "set_per_process_memory_fraction",
        lambda fraction, device: calls.append((fraction, device)),
    )
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda device: None)
    device = configure_device(name, 4)
    assert device == torch.device("cuda", index)
    assert calls == [(0.5, device)]


def test_cpu_benchmark_does_not_initialize_cuda(monkeypatch):
    def unexpected():
        pytest.fail("CPU benchmark must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected)
    assert configure_device("cpu", 4) == torch.device("cpu")


def test_percentile_uses_nearest_rank():
    assert percentile(list(range(1, 31)), 0.95) == 29
