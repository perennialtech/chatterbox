# Contributing

## Run tests

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run the CPU tests:

```bash
uv run --frozen --extra cpu --no-dev --with pytest python -m pytest -q
```

## Measure conversion latency

Run the benchmark on a CUDA host with the `cuda` extra installed. Supply a checkpoint directory containing `s3gen_meanflow.safetensors`, source speech at least ten seconds long, and a target voice reference. Replace the example paths with your files:

```bash
uv run --frozen --extra cuda --no-dev python scripts/benchmark_latency.py \
  --checkpoint /path/to/checkpoints \
  --source /path/to/source.wav \
  --reference /path/to/target.wav \
  --compile \
  --output artifacts/latency.json
```

The benchmark crops the source to 1, 3, 5, and 10 seconds. It reports first-call latency separately, then measures warmed median and p95 latency over 30 completed calls. For this offline pipeline, completed conversion latency is also time to first audio; it is not an HTTP or end-to-end TTS measurement. File mode includes source loading and reference preparation or cache lookup. Use `--input-mode tensor` to measure 16 kHz tensor input with a prepared reference instead.

Use `--prewarm-target` to preload the target and call the public `warmup()` method before timing. That warmup includes a longer utterance than the timed clips, so this option also exercises the startup memory peak. Its time is reported separately as `prewarm_seconds`.

Peak PyTorch allocated and reserved GPU memory include model loading, warmup, and all measured lengths. They exclude CUDA context and other processes; sample process VRAM separately with `nvidia-smi` when comparing a deployment memory budget. The default allocator cap is 5.5 GiB. `--max-memory-gib` changes the cap, but does not reserve memory or protect other workloads from GPU contention.

Keep the device, software versions, thread count, source/reference audio, compiler-cache state, and other GPU activity consistent across comparisons. See `--help` for benchmark controls. Use a separate environment or a read-only copy of each revision when comparing branches; do not modify a running service to benchmark it.

See the [voice conversion performance report](docs/performance.md) for the evaluation method, results, and limits.

## Check flow precision

On CUDA, compare FP32 and FP16 flow execution with fixed noise, identical source tokens, and the same vocoder:

```bash
uv run --frozen --extra cuda --no-dev python scripts/compare_flow_precision.py \
  --checkpoint /path/to/checkpoints \
  --source /path/to/source.wav \
  --reference /path/to/target.wav \
  --compile \
  --output artifacts/precision.json
```

Repeat `--source` to include several speakers or recording styles. The script checks exact output lengths, finite waveforms, mel relative L2 error, and speaker-vector similarity. It includes a clip just above ten seconds to exercise multiple flow chunks when source audio is long enough. These numeric checks do not replace listening tests or a broad speech-quality evaluation.
