from __future__ import annotations

import argparse
from pathlib import Path

from .builder import build_engines
from .config import TrtBuildConfig


def parse_args():
    parser = argparse.ArgumentParser(prog="python -m chatterbox.tensorrt")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--artifact-dir", required=True, type=Path)
    build.add_argument("--output-dir", type=Path)
    build.add_argument("--onnx-precision", choices=["fp32", "fp16"], default="fp32")
    build.add_argument("--engine-precision", choices=["fp32", "fp16"], default="fp16")
    build.add_argument("--workspace-gb", type=float, default=4.0)
    build.add_argument("--shape-plan", type=Path)
    build.add_argument(
        "--strongly-typed",
        action="store_true",
        help="Build a strongly typed TensorRT network using the ONNX tensor dtypes.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        config = TrtBuildConfig(
            artifact_dir=args.artifact_dir,
            output_dir=args.output_dir,
            onnx_precision=args.onnx_precision,
            engine_precision=args.engine_precision,
            workspace_bytes=int(args.workspace_gb * 1024**3),
            shape_plan=args.shape_plan,
            strongly_typed=args.strongly_typed,
        )
        records = build_engines(config)
        print(
            f"Built {len(records)} TensorRT engines under {config.resolved_output_dir}"
        )


if __name__ == "__main__":
    main()
