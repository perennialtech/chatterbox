from __future__ import annotations

import argparse
from pathlib import Path

from chatterbox.tensorrt import TrtBuildConfig, build_engines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TensorRT engines from ONNX artifacts."
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--onnx-precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--engine-precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--workspace-gb", type=float, default=4.0)
    parser.add_argument(
        "--strongly-typed",
        action="store_true",
        help="Build a strongly typed TensorRT network using the ONNX tensor dtypes.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrtBuildConfig(
        artifact_dir=args.artifact_dir,
        output_dir=args.output_dir,
        onnx_precision=args.onnx_precision,
        engine_precision=args.engine_precision,
        workspace_bytes=int(args.workspace_gb * 1024**3),
        strongly_typed=args.strongly_typed,
    )
    records = build_engines(config)
    print(f"Built {len(records)} TensorRT engines under {config.resolved_output_dir}")


if __name__ == "__main__":
    main()
