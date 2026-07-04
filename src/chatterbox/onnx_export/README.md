# Chatterbox VC ONNX export

This package exports stable tensor-only neural subgraphs for the VC-only fork. The high-level Python service remains orchestration code; ONNX graphs do not accept file paths, dictionaries, devices, optional host objects, or raw Python control-flow flags.

## Profiles

- `vc_full_tensor`: exposes tensor-level frontend preprocessing graphs.
- `vc_bucketed`: exports static-shape bucket variants for production runtime.

## Graph signatures

### `s3_tokenizer_quantizer.onnx`

Inputs:

- `log_mel`: `float32[B, n_mels, T_mel_16k]`
- `mel_lengths`: `int64[B]`

Outputs:

- `speech_tokens`: `int64[B, T_token]`
- `speech_token_lengths`: `int64[B]`

### `speaker_encoder.onnx`

Inputs:

- `fbank`: `float32[B, T, 80]`
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

Outputs:

- `mu`: `float[B, 80, 2 * (P + N)]`
- `mask`: `float[B, 1, 2 * (P + N)]`
- `prompt_mel_len`: `int64[B]`
- `output_mel_len`: `int64[B]`

### `conditional_decoder_step.onnx`

Inputs:

- `x`: `float[B, 80, T]`
- `mask`: `float[B, 1, T]`
- `mu`: `float[B, 80, T]`
- `spks`: `float[B, 80]`
- `cond`: `float[B, 80, T]`
- `t`: `float[B]`
- `r`: `float[B]`

Outputs:

- `dxdt`: `float[B, 80, T]`

### `flow_decoder_meanflow2.onnx`

Inputs:

- `noise`: `float[B, 80, T]`
- `mask`: `float[B, 1, T]`
- `mu`: `float[B, 80, T]`
- `spks`: `float[B, 80]`
- `cond`: `float[B, 80, T]`

Outputs:

- `mel`: `float[B, 80, T]`

### `vocoder_hift.onnx`

Inputs:

- `speech_feat`: `float[B, 80, T_mel]`
- `source_phase`: `float[B, 9, 1]`
- `source_noise`: `float[B, 9, T_audio_source]`

Outputs:

- `wav`: `float[B, samples]`
- `source`: `float[B, 1, samples]`

## Artifact layout

```text
out/
  manifest.json
  metadata.json
  fp32/
    token_to_mu.onnx
    conditional_decoder_step.onnx
    flow_decoder_meanflow2.onnx
    vocoder_hift.onnx
  fp16/
  validation/
```

The manifest records checkpoint hash, opset, profile, precision, quantization mode, dynamic axes, sample rates, hop sizes, token rate, vocabulary size, prompt limits, and bucket sizes.

## Runtime

`onnx_export/runtime/vc.py` performs token bucketing, token padding, decoder condition construction, optional host-side Euler stepping, mel cropping, vocoding, and trim fade. File IO remains outside the ONNX runtime.

## Validation thresholds

- tokenizer tokens: exact match
- speaker embedding: cosine similarity `>= 0.999`
- token-to-mu fp32: `max_abs <= 1e-4`, `mean_abs <= 1e-5`
- conditional decoder step fp32: `max_abs <= 2e-3`, `mean_abs <= 2e-4`
- conditional decoder step fp16: `max_abs <= 2e-2`, `mean_abs <= 2e-3`
- vocoder fp32: waveform `mean_abs <= 1e-4`, length exact
- vocoder fp16: waveform `mean_abs <= 5e-3`, length exact

## CLI

```bash
python -m chatterbox.onnx_export export \
  --checkpoint-dir /path/to/chatterbox-turbo \
  --output-dir ./onnx-out \
  --profile vc_full_tensor \
  --precision fp32 \
  --opset 18

python -m chatterbox.onnx_export validate \
  --artifacts ./onnx-out/fp32 \
  --checkpoint-dir /path/to/chatterbox-turbo
```

## TODO

- Validate exported fp32 artifacts on the production checkpoint with ONNX Runtime and store parity reports.
- Enable fp16 conversion in the CLI only after fp32 parity reports pass.
