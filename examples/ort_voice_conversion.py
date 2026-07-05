from __future__ import annotations

import argparse
from pathlib import Path

import torchaudio as ta

from chatterbox import ChatterboxVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Chatterbox VC with exported ONNX Runtime artifacts."
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime provider list.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    providers = [provider.strip() for provider in args.providers.split(",") if provider]

    vc = ChatterboxVC.from_onnx_artifacts(
        args.artifact_dir,
        precision=args.precision,
        providers=providers,
    )
    wav, sample_rate, timings = vc.generate(args.source, target_voice_path=args.target)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(args.output), wav.cpu(), sample_rate)
    print(
        f"wrote={args.output} sample_rate={sample_rate} "
        f"duration={wav.shape[-1] / sample_rate:.3f}s rtf={timings.get('rtf', 0):.3f}"
    )


if __name__ == "__main__":
    main()
