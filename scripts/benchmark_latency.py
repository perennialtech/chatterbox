"""Measure completed conversion latency (also TTFA for this offline pipeline)."""

import argparse
import json
import math
import os
import platform
import statistics
import tempfile
import time
from pathlib import Path

import torch
import torchaudio

from chatterbox.audio import S3_SR, load_audio_mono
from chatterbox.vc.backends.torch_backend import TorchVCBackend


def percentile(values, fraction):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * fraction) - 1)]


def configure_device(name, memory_gib):
    device = torch.device(name)
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        total = torch.cuda.get_device_properties(device).total_memory
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, memory_gib * 2**30 / total), device
        )
        torch.cuda.reset_peak_memory_stats(device)
    return device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--prewarm-target",
        action="store_true",
        help="Prepare the target and warm representative flow/vocoder shapes before timing",
    )
    parser.add_argument("--input-mode", choices=["path", "tensor"], default="path")
    parser.add_argument("--seconds", type=float, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-memory-gib", type=float, default=5.5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (
        args.runs < 1
        or args.warmup < 0
        or args.threads < 1
        or not math.isfinite(args.max_memory_gib)
        or args.max_memory_gib <= 0
        or any(not math.isfinite(s) or s <= 0 for s in args.seconds)
    ):
        parser.error(
            "runs, threads, memory and seconds must be positive; warmup must be non-negative"
        )

    torch.set_num_threads(args.threads)
    device = configure_device(args.device, args.max_memory_gib)

    start = time.perf_counter()
    backend = TorchVCBackend.from_local(
        args.checkpoint, args.device, compile=args.compile
    )
    load_seconds = time.perf_counter() - start
    source = load_audio_mono(args.source, S3_SR, "cpu")
    if source.numel() < round(max(args.seconds) * S3_SR):
        parser.error(
            "source audio must be at least as long as the longest requested clip"
        )

    prewarm_seconds = None
    if args.prewarm_target:
        start = time.perf_counter()
        backend.set_target_voice_path(args.reference)
        backend.warmup()
        prewarm_seconds = time.perf_counter() - start

    report = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "device": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "compile": args.compile,
        "flow_fp16": device.type == "cuda"
        and os.getenv("CHATTERBOX_FLOW_FP16", "1").lower()
        in ("1", "true", "yes", "on"),
        "cuda_graphs": getattr(backend.s3gen, "_use_cuda_graphs", False),
        "input_mode": args.input_mode,
        "threads": args.threads,
        "runs": args.runs,
        "source": str(args.source),
        "reference": str(args.reference),
        "load_seconds": load_seconds,
        "prewarm_seconds": prewarm_seconds,
        "results": [],
    }
    with tempfile.TemporaryDirectory(prefix="chatterbox-bench-") as directory:
        for seconds in args.seconds:
            audio = source[: round(seconds * S3_SR)].unsqueeze(0)
            path = Path(directory) / "source.wav"
            torchaudio.save(str(path), audio, S3_SR)

            def convert():
                if args.input_mode == "path":
                    return backend.convert_from_path(
                        path, target_voice_path=args.reference
                    )
                return backend.convert_from_tensors(audio)

            if args.input_mode == "tensor":
                # Always use the requested voice, even when conds.pt is present.
                backend.convert_from_path(path, target_voice_path=args.reference)

            start = time.perf_counter()
            result = convert()
            first_ms = (time.perf_counter() - start) * 1000
            for _ in range(args.warmup):
                convert()
            latencies = []
            for _ in range(args.runs):
                start = time.perf_counter()
                result = convert()
                latencies.append((time.perf_counter() - start) * 1000)
            if not torch.isfinite(result.wav).all() or result.wav.numel() == 0:
                raise RuntimeError("conversion returned invalid audio")
            row = {
                "input_seconds": seconds,
                "first_call_ms": first_ms,
                "median_ms": statistics.median(latencies),
                "p95_ms": percentile(latencies, 0.95),
                "output_samples": result.wav.numel(),
                "sample_rate": result.sample_rate,
            }
            report["results"].append(row)
            print(json.dumps(row), flush=True)

    if device.type == "cuda":
        report["peak_allocated_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
        report["peak_reserved_gib"] = torch.cuda.max_memory_reserved(device) / 2**30
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
