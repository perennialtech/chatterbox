from __future__ import annotations

import argparse
import shutil
import tempfile
import uuid
from pathlib import Path

from .artifacts import ArtifactRecord, write_manifest
from .config import ExportConfig
from .export_session import ExportSession
from .graph_spec import ExportContext, GraphSpec
from .graphs import ALL_GRAPHS
from .model_loading import load_torch_model, required_export_positional_frames
from .validation.runner import run_validation, run_validation_with_model


def export(config: ExportConfig) -> None:
    final_config = ExportConfig(
        checkpoint_dir=config.checkpoint_dir.resolve(),
        output_dir=config.output_dir.resolve(),
        opset=config.opset,
        external_data=config.external_data,
        validate=config.validate,
        device=config.device,
    )

    staging_dir = _make_staging_dir(final_config.output_dir)
    try:
        staging_config = ExportConfig(
            checkpoint_dir=final_config.checkpoint_dir,
            output_dir=staging_dir,
            opset=final_config.opset,
            external_data=final_config.external_data,
            validate=final_config.validate,
            device=final_config.device,
        )

        model = load_torch_model(
            staging_config.checkpoint_dir,
            device=staging_config.device,
        )
        context = ExportContext.from_model(
            model,
            device=staging_config.device,
        )
        required_positions = required_export_positional_frames()

        session = ExportSession(
            opset=staging_config.opset,
            external_data=staging_config.external_data,
        )
        artifacts: list[ArtifactRecord] = []

        for spec in ALL_GRAPHS:
            module = spec.make_module(model).to(staging_config.device)
            dummy_inputs = spec.make_dummy_inputs(context)
            _assert_dummy_inputs_within_positional_limit(
                spec,
                dummy_inputs,
                required_positions,
            )
            dummy_inputs = tuple(x.to(staging_config.device) for x in dummy_inputs)

            onnx_path = staging_config.onnx_dir / spec.filename
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
            staging_config.output_dir,
            staging_config,
            ALL_GRAPHS,
            artifacts,
            source_hop=int(model.mel2wav.source_hop),
            token_mel_ratio=int(model._token_mel_ratio),
            final_context_token_count=int(model._final_context_token_count),
            vocoder_harmonics=int(model.mel2wav.nb_harmonics + 1),
        )

        if staging_config.validate:
            run_validation_with_model(
                artifact_dir=staging_config.output_dir,
                model=model,
                device=staging_config.device,
            )

        _replace_output_dir(staging_dir, final_config.output_dir)
        staging_dir = None
    finally:
        if staging_dir is not None and staging_dir.exists():
            shutil.rmtree(staging_dir)


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


def _make_staging_dir(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.",
            suffix=".tmp",
            dir=output_dir.parent,
        )
    ).resolve()


def _replace_output_dir(staging_dir: Path, output_dir: Path) -> None:
    staging_dir = staging_dir.resolve()
    output_dir = output_dir.resolve()
    backup_dir: Path | None = None

    if output_dir.exists():
        backup_dir = output_dir.with_name(f".{output_dir.name}.old-{uuid.uuid4().hex}")
        output_dir.replace(backup_dir)

    try:
        staging_dir.replace(output_dir)
    except Exception:
        if backup_dir is not None and backup_dir.exists() and not output_dir.exists():
            backup_dir.replace(output_dir)
        raise

    if backup_dir is not None and backup_dir.exists():
        shutil.rmtree(backup_dir)


def _assert_dummy_inputs_within_positional_limit(
    spec: GraphSpec,
    inputs: tuple,
    max_positions: int,
) -> None:
    length = _dummy_positional_length(spec.input_names, inputs)
    if length > max_positions:
        raise AssertionError(
            f"{spec.name} dummy sequence length {length} exceeds prepared "
            f"positional cache length {max_positions}"
        )


def _dummy_positional_length(input_names: list[str], inputs: tuple) -> int:
    max_length = 0
    for name, tensor in zip(input_names, inputs):
        if name == "token" and tensor.ndim >= 2:
            max_length = max(max_length, int(tensor.shape[1]))
        elif name == "fbank" and tensor.ndim >= 2:
            max_length = max(max_length, int(tensor.shape[1]))
        elif name in {"log_mel", "noise", "mask", "mu", "cond", "speech_feat"}:
            if tensor.ndim >= 3:
                max_length = max(max_length, int(tensor.shape[2]))
    return max_length


if __name__ == "__main__":
    main()
