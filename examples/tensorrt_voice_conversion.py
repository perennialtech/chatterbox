from __future__ import annotations

import argparse
from pathlib import Path

import torchaudio as ta

from chatterbox import ChatterboxVC


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Chatterbox VC with TensorRT engines."
    )
    parser.add_argument("--engine-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    vc = ChatterboxVC.from_tensorrt_engines(args.engine_dir)
    wav, sample_rate, timings = vc.generate(args.source, target_voice_path=args.target)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(args.output), wav.cpu(), sample_rate)
    print(
        f"wrote={args.output} sample_rate={sample_rate} "
        f"duration={wav.shape[-1] / sample_rate:.3f}s rtf={timings.get('rtf', 0):.3f}"
    )


if __name__ == "__main__":
    main()
