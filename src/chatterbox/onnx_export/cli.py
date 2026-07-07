from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import ArtifactRecord, write_manifest
from .config import ExportConfig
from .export_session import ExportSession
from .graphs import ALL_GRAPHS
from .model_loading import load_torch_model
from .precision import convert_fp16
from .validation.runner import run_validation_for_precisions


def export(config: ExportConfig) -> None:
    model = load_torch_model(
        config.checkpoint_dir,
        device=config.device,
        max_positions=config.max_positional_frames,
    )
    session = ExportSession(opset=config.opset, external_data=config.external_data)
    artifacts: list[ArtifactRecord] = []

    fp32_dir = config.onnx_precision_dir("fp32")
    for spec in ALL_GRAPHS:
        module = spec.make_module(model).to(config.device)
        if spec.name == "vocoder_hift":
            from .graphs.vocoder import make_model_dummy_inputs

            dummy_inputs = tuple(
                x.to(config.device) for x in make_model_dummy_inputs(model)
            )
        else:
            dummy_inputs = tuple(x.to(config.device) for x in spec.make_dummy_inputs())

        fp32_path = fp32_dir / spec.filename
        artifacts.append(
            session.export(
                graph_name=spec.name,
                precision="fp32",
                module=module,
                path=fp32_path,
                inputs=dummy_inputs,
                input_names=spec.input_names,
                output_names=spec.output_names,
                dynamic_shapes=spec.dynamic_shapes,
            )
        )

        if "fp16" in config.precisions:
            fp16_path = config.onnx_precision_dir("fp16") / spec.filename
            convert_fp16(fp32_path, fp16_path, external_data=config.external_data)
            artifacts.append(
                ArtifactRecord(
                    graph_name=spec.name,
                    precision="fp16",
                    path=str(fp16_path),
                    inputs=spec.input_names,
                    outputs=spec.output_names,
                    dynamic_shapes=spec.dynamic_shapes,
                )
            )

    write_manifest(
        config.output_dir,
        config,
        ALL_GRAPHS,
        artifacts,
        source_hop=int(model.mel2wav.source_hop),
    )

    if config.validate:
        run_validation_for_precisions(
            artifact_dir=config.output_dir,
            checkpoint_dir=config.checkpoint_dir,
            precisions=config.precisions,
            device=config.device,
        )


def validate_artifacts(
    artifact_dir: Path,
    checkpoint_dir: Path,
    precision: str,
    device: str,
) -> None:
    config = ExportConfig(
        checkpoint_dir=checkpoint_dir,
        output_dir=artifact_dir,
        precision=precision,  # type: ignore[arg-type]
        device=device,
    )
    run_validation_for_precisions(
        artifact_dir=artifact_dir,
        checkpoint_dir=checkpoint_dir,
        precisions=config.precisions,
        device=device,
    )
    print(f"Validated ONNX parity for {precision} artifacts under {artifact_dir}")


def parse_args():
    parser = argparse.ArgumentParser(prog="python -m chatterbox.onnx_export")
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export")
    export_p.add_argument("--checkpoint-dir", required=True, type=Path)
    export_p.add_argument("--output-dir", required=True, type=Path)
    export_p.add_argument(
        "--precision", default="fp32", choices=["fp32", "fp16", "both"]
    )
    export_p.add_argument("--opset", default=18, type=int)
    export_p.add_argument("--device", default="cpu")
    export_p.add_argument(
        "--external-data", action=argparse.BooleanOptionalAction, default=True
    )
    export_p.add_argument(
        "--validate", action=argparse.BooleanOptionalAction, default=True
    )

    val_p = sub.add_parser("validate")
    val_p.add_argument("--artifact-dir", required=True, type=Path)
    val_p.add_argument("--checkpoint-dir", required=True, type=Path)
    val_p.add_argument("--precision", default="fp32", choices=["fp32", "fp16", "both"])
    val_p.add_argument("--device", default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        config = ExportConfig(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            precision=args.precision,
            opset=args.opset,
            validate=args.validate,
            external_data=args.external_data,
            device=args.device,
        )
        export(config)
    elif args.command == "validate":
        validate_artifacts(
            args.artifact_dir, args.checkpoint_dir, args.precision, args.device
        )


if __name__ == "__main__":
    main()
