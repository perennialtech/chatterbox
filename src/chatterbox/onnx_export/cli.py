from __future__ import annotations

import argparse
from pathlib import Path

from .artifacts import ArtifactRecord, write_manifest
from .config import ExportConfig
from .export_session import ExportSession
from .graphs import ALL_GRAPHS
from .model_loading import load_torch_model
from .validation.runner import run_validation, run_validation_with_model


def export(config: ExportConfig) -> None:
    config = ExportConfig(
        checkpoint_dir=config.checkpoint_dir.resolve(),
        output_dir=config.output_dir.resolve(),
        opset=config.opset,
        external_data=config.external_data,
        validate=config.validate,
        device=config.device,
        max_positional_frames=config.max_positional_frames,
    )

    model = load_torch_model(
        config.checkpoint_dir,
        device=config.device,
        max_positions=config.max_positional_frames,
    )
    session = ExportSession(opset=config.opset, external_data=config.external_data)
    artifacts: list[ArtifactRecord] = []

    for spec in ALL_GRAPHS:
        module = spec.make_module(model).to(config.device)
        dummy_inputs = tuple(x.to(config.device) for x in spec.make_dummy_inputs())

        onnx_path = config.onnx_dir / spec.filename
        artifacts.append(
            session.export(
                graph_name=spec.name,
                module=module,
                path=onnx_path,
                inputs=dummy_inputs,
                input_names=spec.input_names,
                output_names=spec.output_names,
                dynamic_shapes=spec.dynamic_shapes,
            )
        )

    write_manifest(
        config.output_dir,
        config,
        ALL_GRAPHS,
        artifacts,
        source_hop=int(model.mel2wav.source_hop),
        token_mel_ratio=int(model._token_mel_ratio),
        final_context_token_count=int(model._final_context_token_count),
        vocoder_harmonics=int(model.mel2wav.nb_harmonics + 1),
    )

    if config.validate:
        run_validation_with_model(
            artifact_dir=config.output_dir,
            model=model,
            device=config.device,
        )


def validate_artifacts(
    artifact_dir: Path,
    checkpoint_dir: Path,
    device: str,
) -> None:
    run_validation(
        artifact_dir=artifact_dir.resolve(),
        checkpoint_dir=checkpoint_dir.resolve(),
        device=device,
    )
    print(f"Validated ONNX parity under {artifact_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m chatterbox.onnx_export")
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export")
    export_p.add_argument("--checkpoint-dir", required=True, type=Path)
    export_p.add_argument("--output-dir", required=True, type=Path)
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
    val_p.add_argument("--device", default="cpu")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        config = ExportConfig(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            opset=args.opset,
            validate=args.validate,
            external_data=args.external_data,
            device=args.device,
        )
        export(config)
    elif args.command == "validate":
        validate_artifacts(args.artifact_dir, args.checkpoint_dir, args.device)


if __name__ == "__main__":
    main()
