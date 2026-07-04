# Chatterbox VC ONNX and TensorRT export

The exporter emits the complete tensor-level VC graph set for deterministic service orchestration. File IO, audio loading, target voice caching, Euler orchestration choices, and backend selection remain outside the ONNX graphs.

## Export

```bash
uv run --extra cuda,tensorrt,onnx python -m chatterbox.onnx_export export \
  --checkpoint-dir checkpoints \
  --output-dir ./artifacts \
  --precision both \
  --opset 18 \
  --device cuda \
  --validate
```

Supported ONNX precisions:

- `fp32`
- `fp16`
- `both`

FP16 ONNX artifacts keep FP32 graph inputs and outputs while converting eligible internal tensors and initializers to FP16.

## Validation

```bash
uv run python -m chatterbox.onnx_export validate \
  --artifact-dir ./artifacts \
  --checkpoint-dir checkpoints \
  --precision both \
  --device cuda
```

Validation runs graph-by-graph parity against Torch with deterministic dummy tensors and writes reports to:

```text
artifacts/validation/fp32.json
artifacts/validation/fp16.json
```

Tokenizer outputs are checked exactly. Speaker embeddings use cosine similarity. Neural float outputs use graph-specific absolute-difference tolerances.

## Artifact layout

```text
artifacts/
  manifest.json
  metadata.json
  onnx/
    fp32/
      s3_tokenizer_quantizer.onnx
      speaker_encoder.onnx
      reference_mel_24k.onnx
      token_to_mu.onnx
      conditional_decoder_step.onnx
      flow_decoder_meanflow2.onnx
      vocoder_hift.onnx
    fp16/
      s3_tokenizer_quantizer.onnx
      speaker_encoder.onnx
      reference_mel_24k.onnx
      token_to_mu.onnx
      conditional_decoder_step.onnx
      flow_decoder_meanflow2.onnx
      vocoder_hift.onnx
  validation/
    fp32.json
    fp16.json
```

The root manifest uses schema version `2` and records checkpoint hash, opset, exported precisions, graph signatures, dtype expectations, dynamic axes, sample rates, hop sizes, token rate, vocabulary size, source hop, meanflow schedule, and trim-fade length.

## Graph signatures

### `s3_tokenizer_quantizer.onnx`

Inputs:

- `log_mel`: `float32[B, 128, T_mel_16k]`
- `mel_lengths`: `int64[B]`

Outputs:

- `speech_tokens`: `int64[B, T_token]`
- `speech_token_lengths`: `int64[B]`

### `speaker_encoder.onnx`

Inputs:

- `fbank`: `float32[B, T_fbank, 80]`
- `fbank_lengths`: `int64[B]`

Outputs:

- `embedding`: `float32[B, 192]`

### `reference_mel_24k.onnx`

Inputs:

- `wav_24k`: `float32[B, samples]`

Outputs:

- `prompt_feat`: `float32[B, T_mel, 80]`
- `prompt_feat_len`: `int64[B]`

### `token_to_mu.onnx`

Inputs:

- `prompt_token`: `int64[B, P]`
- `prompt_token_len`: `int64[B]`
- `speech_token`: `int64[B, N]`
- `speech_token_len`: `int64[B]`
- `embedding`: `float32[B, 192]`

Outputs:

- `mu`: `float32[B, 80, 2 * (P + N)]`
- `mask`: `float32[B, 1, 2 * (P + N)]`
- `spks`: `float32[B, 80]`
- `prompt_mel_len`: `int64[B]`
- `output_mel_len`: `int64[B]`

### `conditional_decoder_step.onnx`

Inputs:

- `x`: `float32[B, 80, T]`
- `mask`: `float32[B, 1, T]`
- `mu`: `float32[B, 80, T]`
- `spks`: `float32[B, 80]`
- `cond`: `float32[B, 80, T]`
- `t`: `float32[B]`
- `r`: `float32[B]`

Outputs:

- `dxdt`: `float32[B, 80, T]`

### `flow_decoder_meanflow2.onnx`

Inputs:

- `noise`: `float32[B, 80, T]`
- `mask`: `float32[B, 1, T]`
- `mu`: `float32[B, 80, T]`
- `spks`: `float32[B, 80]`
- `cond`: `float32[B, 80, T]`

Outputs:

- `mel`: `float32[B, 80, T]`

### `vocoder_hift.onnx`

Inputs:

- `speech_feat`: `float32[B, 80, T_mel]`
- `source_phase`: `float32[B, 9, 1]`
- `source_noise`: `float32[B, 9, T_mel * source_hop]`

Outputs:

- `wav`: `float32[B, samples]`
- `source`: `float32[B, 1, samples]`

## TensorRT build

Build TensorRT engines from exported ONNX artifacts:

```bash
uv run --extra cuda,tensorrt,onnx python -m chatterbox.tensorrt build \
  --artifact-dir ./artifacts \
  --onnx-precision fp32 \
  --engine-precision fp32 \
  --workspace-gb 4
```

Output:

```text
artifacts/
  tensorrt/
    fp16/
      trt_manifest.json
      s3_tokenizer_quantizer.engine
      speaker_encoder.engine
      reference_mel_24k.engine
      token_to_mu.engine
      conditional_decoder_step.engine
      flow_decoder_meanflow2.engine
      vocoder_hift.engine
```

## TensorRT shape plan

TensorRT builds require optimization profiles. Defaults target batch size `1` and up to `3072` total tokens / `6144` mel frames.

Override ranges with JSON:

```json
{
  "graphs": {
    "flow_decoder_meanflow2": {
      "noise": {
        "min": [1, 80, 2],
        "opt": [1, 80, 512],
        "max": [1, 80, 4096]
      }
    }
  }
}
```

Build with:

```bash
uv run python -m chatterbox.tensorrt build \
  --artifact-dir ./artifacts \
  --shape-plan ./shape-plan.json
```

## Runtime examples

ONNX Runtime:

```python
from chatterbox import ChatterboxVC

vc = ChatterboxVC.from_onnx_artifacts("./artifacts", precision="fp32")
wav, sr, timings = vc.generate("source.wav", target_voice_path="target.wav")
```

Native TensorRT:

```python
from chatterbox import ChatterboxVC

vc = ChatterboxVC.from_tensorrt_engines("./artifacts/tensorrt/fp16")
wav, sr, timings = vc.generate("source.wav", target_voice_path="target.wav")
```

The ONNX and TensorRT VC backends require all seven exported graphs. Missing artifacts or engines fail at backend construction or first use with explicit backend errors.
