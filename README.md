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
# PyTorch execution
uv sync

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

Target voice tensors can be cached and reused:

```python
vc.set_target_voice_from_tensors(condition_tensors)
wav, sample_rate, timings = vc.generate(source_audio_16k_tensor)
```

## Gradio app

The repository includes a browser UI in [`app.py`](app.py). Install the `ui` extra together with the runtime extras for the backend you plan to use, then launch:

```bash
uv run python app.py --backend pytorch --device cuda
```

The app accepts source speech and target-voice reference audio, and returns playable/downloadable converted audio plus timing details. Use `--server-name 0.0.0.0` when serving from a container or remote host, and `uv run python app.py --help` for the full launch options.

## Determinism

The flow decoder starts from noise, and the vocoder source path uses phase/noise inputs. High-level backends generate these tensors internally, so repeated runs are not bit-identical.
