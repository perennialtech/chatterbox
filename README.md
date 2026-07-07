# Chatterbox

Chatterbox is a voice conversion pipeline. This package provides the PyTorch implementation, exports the tensor-level pipeline to ONNX, and can build TensorRT engines from those ONNX artifacts. The exported artifacts are intended for service runtimes that want to orchestrate preprocessing, reference-voice caching, flow sampling, and vocoding outside PyTorch.

The stable contract for an ONNX artifact directory is `manifest.json`. Use it as the source of truth for graph files, input/output names, dtypes, dynamic axes, runtime constants, and precision availability.

See [`examples/`](examples/) for runnable integrations.

## Scope

The package contains:

- PyTorch implementation of the high-level VC pipeline
- `ChatterboxVC` inference backends for PyTorch, ONNX, and TensorRT
- checkpoint loading for the meanflow S3 generation model
- ONNX export for the graph set used by the VC runtime
- optional FP16 ONNX conversion with FP32 graph inputs and outputs
- graph-by-graph ONNX Runtime parity validation
- TensorRT engine building from exported ONNX graphs
- low-level graph runners for custom orchestration

The exported graphs do not include audio file I/O. Runtime integrations are responsible for:

- loading audio
- resampling to the model sample rates
- computing source log-mel features for tokenization
- computing target speaker fbank features
- caching target-voice conditioning tensors when desired
- providing deterministic noise/phase tensors when deterministic output is required
- choosing ONNX Runtime providers or TensorRT engines

The high-level `ChatterboxVC` backends handle this orchestration for common local use.

## Installation

Install the project with the runtime extras needed by your workflow.

```bash
# PyTorch execution
uv sync

# Gradio UI with CPU PyTorch execution
uv sync --extra cpu,ui

# ONNX export and ONNX Runtime execution
uv sync --extra cpu,onnx

# ONNX Runtime with the Gradio UI
uv sync --extra cpu,onnx,ui

# CUDA PyTorch execution, export, plus ONNX/TensorRT tooling
uv sync --extra cuda,onnx,tensorrt

# CUDA execution with the Gradio UI
uv sync --extra cuda,onnx,tensorrt,ui
```

TensorRT builds require a compatible NVIDIA driver, CUDA runtime, TensorRT Python package, and `cuda-python`. Use `python -m chatterbox.tensorrt build --help` and TensorRT's own diagnostics when validating a deployment image.

## Checkpoints

The exporter expects a checkpoint directory compatible with `chatterbox.onnx_export.model_loading.load_torch_model`. For the shipped loader this directory must contain the S3Gen meanflow checkpoint file used by that function.

The Torch VC backend can also consume the optional built-in voice-conditioning file when present. ONNX and TensorRT deployments can extract target voice tensors from an audio file at runtime or accept cached tensors through `set_target_voice_from_tensors`.

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

## Export

Run the exporter from the repository environment:

```bash
uv run python -m chatterbox.onnx_export export \
  --checkpoint-dir checkpoints \
  --output-dir artifacts \
  --precision both \
  --device cuda \
  --validate
```

Use `--help` for the complete CLI surface:

```bash
uv run python -m chatterbox.onnx_export export --help
```

Export writes an artifact directory with:

- `manifest.json`
- `metadata.json`
- ONNX files grouped by precision
- validation reports when validation is enabled

Do not hard-code the graph inventory or signatures in production code. Read `manifest.json`, or use `OnnxSessions.from_artifact_dir`, which validates required runtime graphs before creating sessions.

## Validation

Validation compares each exported graph against the PyTorch module with deterministic dummy tensors.

```bash
uv run python -m chatterbox.onnx_export validate \
  --artifact-dir artifacts \
  --checkpoint-dir checkpoints \
  --precision both \
  --device cuda
```

Validation reports are written under `artifacts/validation/`. These reports are useful for CI, image qualification, and verifying provider changes.

Tokenizer outputs are exact checks. Speaker embeddings use cosine similarity. Floating neural outputs use graph-specific absolute-difference tolerances from the validation package.

## ONNX Runtime usage

High-level voice conversion using the ONNX backend:

```python
from chatterbox import ChatterboxVC

vc = ChatterboxVC.from_onnx_artifacts(
    "artifacts",
    precision="fp32",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

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

For custom orchestration, use `OnnxSessions` and graph runners:

```python
from chatterbox.onnx_export.runtime import OnnxSessions

sessions = OnnxSessions.from_artifact_dir(
    "artifacts",
    precision="fp32",
    providers=["CPUExecutionProvider"],
)

runner = sessions.runner("s3_tokenizer_quantizer")
outputs = runner.run({"log_mel": log_mel, "mel_lengths": mel_lengths})
```

The manifest describes the exact graph names and tensor contracts. The example `manual_onnx_pipeline.py` shows end-to-end graph-level orchestration without `ChatterboxVC`.

## TensorRT build

Build TensorRT engines from an exported artifact directory:

```bash
uv run python -m chatterbox.tensorrt build \
  --artifact-dir artifacts \
  --workspace-gb 4
```

By default, engines are written under the artifact directory's TensorRT output tree. Pass `--output-dir` to write elsewhere.

The builder reads graph metadata from the ONNX manifest and writes a TensorRT manifest next to the engines. Runtime construction uses that TensorRT manifest.

## TensorRT shape profiles

TensorRT requires optimization profiles for dynamic inputs. The builder provides defaults suitable for the packaged VC runtime. Override them with a JSON shape plan when your deployment has different limits.

Shape-plan files are keyed by graph name and input name:

```json
{
  "graphs": {
    "flow_decoder_meanflow2": {
      "noise": {
        "min": [1, 80, 2],
        "opt": [1, 80, 768],
        "max": [1, 80, 4096]
      }
    }
  }
}
```

Build with:

```bash
uv run python -m chatterbox.tensorrt build \
  --artifact-dir artifacts \
  --shape-plan shape-plan.json
```

The loader validates rank, range ordering, and the runtime batch-size constraint before building. The default profile definitions live in code; prefer reading or overriding them there instead of copying them into documentation.

## TensorRT runtime usage

High-level voice conversion using the TensorRT backend:

```python
from chatterbox import ChatterboxVC

vc = ChatterboxVC.from_tensorrt_engines("artifacts/tensorrt")
wav, sample_rate, timings = vc.generate(
    "source.wav",
    target_voice_path="target.wav",
)
```

The TensorRT backend accepts the same cached target-voice tensors as the ONNX backend.

## Gradio app

The repository includes a browser UI in [`app.py`](app.py). Install the `ui` extra together with the runtime extras for the backend you plan to use, then launch one of:

```bash
uv run python app.py --backend pytorch --device cuda
uv run python app.py --backend onnx --onnx-artifact-dir artifacts --onnx-providers CUDAExecutionProvider,CPUExecutionProvider
uv run python app.py --backend tensorrt --tensorrt-engine-dir artifacts/tensorrt
```

The app accepts source speech and target-voice reference audio, and returns playable/downloadable converted audio plus timing details. Use `--server-name 0.0.0.0` when serving from a container or remote host, and `uv run python app.py --help` for the full launch options.

## Programmatic export

```python
from pathlib import Path

from chatterbox.onnx_export.cli import export
from chatterbox.onnx_export.config import ExportConfig

config = ExportConfig(
    checkpoint_dir=Path("checkpoints"),
    output_dir=Path("artifacts"),
    precision="both",
    device="cuda",
    validate=True,
)

export(config)
```

For a full script, see `examples/export_artifacts.py`.

## Programmatic TensorRT build

```python
from pathlib import Path

from chatterbox.tensorrt import TrtBuildConfig, build_engines

config = TrtBuildConfig(
    artifact_dir=Path("artifacts"),
    workspace_bytes=4 * 1024**3,
)

build_engines(config)
```

For a full script, see `examples/build_tensorrt_engines.py`.

## Artifact consumption rules

For robust downstream integrations:

1. Treat `manifest.json` as the artifact contract.
2. Use graph input names from the manifest or `OnnxSessions.runner(...)`.
3. Feed arrays with the declared dtype and contiguous memory layout.
4. Keep target-voice conditioning tensors in the format accepted by `VoiceConditionTensors`.
5. Use the manifest constants for source-hop and trim-fade behavior.
6. Seed and pass noise tensors explicitly when deterministic conversion is required.
7. Validate exported artifacts in the same environment class used for deployment.

The high-level backends already follow these rules. Custom runtimes should copy the orchestration pattern from the examples rather than relying on incidental implementation details.

## Determinism

The flow decoder starts from noise, and the vocoder source path uses phase/noise inputs. High-level backends generate these tensors internally, so repeated runs are not bit-identical. For deterministic graph-level execution, provide fixed `noise`, `source_phase`, and `source_noise` tensors as shown in the manual ONNX example.

## Examples

The [`examples/`](examples/) directory contains:

- `export_artifacts.py` — programmatic export and optional validation
- `ort_voice_conversion.py` — high-level ONNX Runtime VC integration
- `tensorrt_voice_conversion.py` — high-level TensorRT VC integration
- `build_tensorrt_engines.py` — programmatic TensorRT engine build
- `manual_onnx_pipeline.py` — low-level ONNX graph orchestration with explicit deterministic tensors

Each example has `--help` output and is intended to be copied into downstream projects as a starting point.
