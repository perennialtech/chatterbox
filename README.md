# Chatterbox

Chatterbox is a voice conversion pipeline. This package provides the PyTorch implementation.

## Scope

The package contains:

- PyTorch implementation of the high-level VC pipeline
- `ChatterboxVC` inference backend for PyTorch
- checkpoint loading for the meanflow S3 generation model

## Installation

Install the project with the runtime extras needed by your workflow.

```bash
# CPU PyTorch execution
uv sync --extra cpu

# Gradio UI with CPU PyTorch execution
uv sync --extra cpu,ui

# CUDA PyTorch execution
uv sync --extra cuda

# CUDA execution with the Gradio UI
uv sync --extra cuda,ui
```

## Checkpoints

The Torch VC backend can consume the optional built-in voice-conditioning file when present.

## PyTorch usage

High-level voice conversion using the PyTorch backend:

```python
from chatterbox import ChatterboxVC

vc = ChatterboxVC.from_pretrained(device="cuda")
wav, sample_rate, timings = vc.generate(
    "source.wav",
    target_voice_path="target.wav",
)
```

Prepared target voice conditions can be cached and reused:

```python
vc.set_target_voice_condition(reference_condition)
wav, sample_rate, timings = vc.generate(source_audio_16k_tensor)
```

For a service using one reference file, prepare it before accepting requests:

```python
vc = ChatterboxVC.from_pretrained(device="cuda", compile=True)
vc.set_target_voice_path("target.wav")
vc.warmup()
wav, sample_rate, timings = vc.generate(source_audio_16k_tensor)
```

Tensor inputs must contain mono audio sampled at 16 kHz. File inputs are decoded and resampled automatically. Repeated `target_voice_path` values reuse the active reference while the file is unchanged. Only one reference is cached; selecting another voice replaces it. Changes to the file, including replacement or a symlink pointing to another file, trigger preparation again.

Serialize calls on a shared `ChatterboxVC` instance, including target-voice changes.

Compilation and new input shapes can make initial calls slow. `warmup()` prepares representative flow and vocoder shapes with the active target voice; it does not guarantee that every possible input shape is already compiled.

CUDA conversion uses FP16 autocast for the flow model, with FP32 solver state. Tokenization, reference preparation, and vocoding stay in FP32. Set `CHATTERBOX_FLOW_FP16=0` to use FP32 flow execution. With `compile=True`, the bucketed flow modules also use CUDA graph replay; set `CHATTERBOX_CUDA_GRAPHS=0` to disable replay without disabling compilation. CPU execution does not use these CUDA optimizations.

## Gradio app

The repository includes a browser UI in [`app.py`](app.py). Install the `ui` extra together with the runtime extras for the backend you plan to use, then launch:

```bash
uv run python app.py --backend pytorch --device cuda
```

The app accepts source speech and target-voice reference audio, and returns playable/downloadable converted audio plus timing details. Use `--server-name 0.0.0.0` when serving from a container or remote host, and `uv run python app.py --help` for the full launch options.

## Determinism

The flow decoder starts from noise, and the vocoder source path uses phase/noise inputs. High-level backends generate these tensors internally, so repeated runs are not bit-identical.

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for tests, latency benchmarks, and precision checks.
