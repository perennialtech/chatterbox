from __future__ import annotations

import argparse
from pathlib import Path

from chatterbox.onnx_export.cli import export as export_onnx_artifacts
from chatterbox.onnx_export.config import ExportConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Chatterbox VC ONNX artifacts.")
    parser.add_argument("--checkpoint-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "both"),
        default="fp32",
        help="ONNX precision set to write.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--validate",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run graph parity validation after export.",
    )
    parser.add_argument(
        "--external-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow ONNX external data files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = ExportConfig(
        checkpoint_dir=args.checkpoint_dir,
        output_dir=args.output_dir,
        precision=args.precision,
        device=args.device,
        opset=args.opset,
        validate=args.validate,
        external_data=args.external_data,
    )
    export_onnx_artifacts(config)
    print(f"Wrote ONNX artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
