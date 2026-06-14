import time
from contextlib import contextmanager

import torch


class InferenceTimer:
    def __init__(self, timings=None, device=None, prefix=""):
        self.timings = timings if timings is not None else {}
        self.device = device
        self.prefix = prefix.strip(".")

    def child(self, prefix):
        prefix = prefix.strip(".")
        if self.prefix:
            prefix = f"{self.prefix}.{prefix}"
        return InferenceTimer(self.timings, self.device, prefix)

    def sync(self):
        if self.device is None:
            return

        device_type = str(self.device).lower()
        if "cuda" in device_type and torch.cuda.is_available():
            torch.cuda.synchronize()
        elif (
            "mps" in device_type
            and hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            torch.mps.synchronize()

    def _key(self, key):
        return f"{self.prefix}.{key}" if self.prefix else key

    def record(self, key, value):
        self.timings[self._key(key)] = value

    def add(self, key, value):
        key = self._key(key)
        self.timings[key] = self.timings.get(key, 0.0) + value

    @contextmanager
    def track(self, key):
        key = self._key(key)
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.timings[key] = self.timings.get(key, 0.0) + (
                time.perf_counter() - start
            )


class _NoopTimer:
    def child(self, prefix):
        return self

    def sync(self):
        pass

    def record(self, key, value):
        pass

    def add(self, key, value):
        pass

    @contextmanager
    def track(self, key):
        yield


NOOP_TIMER = _NoopTimer()


def ensure_timer(timer):
    return timer if timer is not None else NOOP_TIMER
