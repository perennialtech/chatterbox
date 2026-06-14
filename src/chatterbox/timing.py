import time
from contextlib import contextmanager

import torch


class InferenceTimer:
    def __init__(self, timings=None, device=None, prefix="", pending_cuda_events=None):
        self._timings = timings if timings is not None else {}
        self.device = device
        self.prefix = prefix.strip(".")
        self._pending_cuda_events = (
            pending_cuda_events if pending_cuda_events is not None else []
        )

    @property
    def timings(self):
        self.finalize()
        return self._timings

    def set_device(self, device):
        if self.device is None:
            self.device = device

    def child(self, prefix):
        prefix = prefix.strip(".")
        if self.prefix:
            prefix = f"{self.prefix}.{prefix}"
        return InferenceTimer(
            self._timings,
            self.device,
            prefix,
            self._pending_cuda_events,
        )

    def _device(self):
        if self.device is None:
            return None
        return torch.device(self.device)

    def _is_cuda(self):
        device = self._device()
        return (
            device is not None and device.type == "cuda" and torch.cuda.is_available()
        )

    def _is_mps(self):
        device = self._device()
        return (
            device is not None
            and device.type == "mps"
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )

    def _key(self, key):
        return f"{self.prefix}.{key}" if self.prefix else key

    def _flush_cuda_events(self):
        if not self._pending_cuda_events:
            return

        pending = list(self._pending_cuda_events)
        self._pending_cuda_events.clear()

        for key, start, end in pending:
            self._timings[key] = self._timings.get(key, 0.0) + (
                start.elapsed_time(end) / 1000.0
            )

    def sync(self):
        if self._is_cuda():
            torch.cuda.synchronize(self._device())
            self._flush_cuda_events()
            return

        if self._is_mps():
            torch.mps.synchronize()

    def finalize(self):
        if self._is_cuda() and self._pending_cuda_events:
            torch.cuda.synchronize(self._device())
            self._flush_cuda_events()

    def record(self, key, value):
        self._timings[self._key(key)] = value

    def add(self, key, value):
        key = self._key(key)
        self._timings[key] = self._timings.get(key, 0.0) + value

    @contextmanager
    def track(self, key):
        key = self._key(key)

        if self._is_cuda():
            device = self._device()
            with torch.cuda.device(device):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)

                start.record()
                try:
                    yield
                finally:
                    end.record()
                    self._pending_cuda_events.append((key, start, end))
            return

        if self._is_mps():
            self.sync()
            start = time.perf_counter()
            try:
                yield
            finally:
                self.sync()
                self._timings[key] = self._timings.get(key, 0.0) + (
                    time.perf_counter() - start
                )
            return

        start = time.perf_counter()
        try:
            yield
        finally:
            self._timings[key] = self._timings.get(key, 0.0) + (
                time.perf_counter() - start
            )


class _NoopTimer:
    def set_device(self, device):
        pass

    def child(self, prefix):
        return self

    def sync(self):
        pass

    def finalize(self):
        pass

    def record(self, key, value):
        pass

    def add(self, key, value):
        pass

    @contextmanager
    def track(self, key):
        yield


NOOP_TIMER = _NoopTimer()


def ensure_timer(timer, device=None):
    if timer is None:
        return NOOP_TIMER

    if device is not None and hasattr(timer, "set_device"):
        timer.set_device(device)

    return timer
