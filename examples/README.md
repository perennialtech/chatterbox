# Chatterbox examples

These scripts are small integration examples for downstream consumers. Run them from an environment where `chatterbox` is importable.

```bash
python examples/export_artifacts.py --help
python examples/ort_voice_conversion.py --help
python examples/tensorrt_voice_conversion.py --help
python examples/build_tensorrt_engines.py --help
python examples/manual_onnx_pipeline.py --help
```

Use the high-level examples when you want the packaged runtime orchestration. Use `manual_onnx_pipeline.py` when implementing your own service runner around the exported graphs.
