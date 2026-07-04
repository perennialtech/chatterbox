import argparse
from pathlib import Path

from safetensors.torch import load_file

from ..models.s3gen import S3Gen
from ..models.s3gen.checkpoint_conversion import \
    convert_diffusers_transformer_keys
from .artifacts import write_manifest
from .config import ExportConfig
from .export_session import ExportSession
from .modules.conditional_decoder import ConditionalDecoderStepExport
from .modules.conditional_decoder import dynamic_axes as decoder_axes
from .modules.conditional_decoder import input_names as decoder_inputs
from .modules.conditional_decoder import make_dummy_inputs as decoder_dummy
from .modules.conditional_decoder import output_names as decoder_outputs
from .modules.flow_decoder import FlowDecoderMeanflow2Export
from .modules.flow_decoder import dynamic_axes as flow_axes
from .modules.flow_decoder import input_names as flow_inputs
from .modules.flow_decoder import make_dummy_inputs as flow_dummy
from .modules.flow_decoder import output_names as flow_outputs
from .modules.token_to_mu import TokenToMuExport
from .modules.token_to_mu import dynamic_axes as token_axes
from .modules.token_to_mu import input_names as token_inputs
from .modules.token_to_mu import make_dummy_inputs as token_dummy
from .modules.token_to_mu import output_names as token_outputs
from .modules.vocoder import VocoderExport
from .modules.vocoder import dynamic_axes as vocoder_axes
from .modules.vocoder import input_names as vocoder_inputs
from .modules.vocoder import make_dummy_inputs as vocoder_dummy
from .modules.vocoder import output_names as vocoder_outputs
from .names import (CONDITIONAL_DECODER_STEP, FLOW_DECODER_MEANFLOW2,
                    TOKEN_TO_MU, VOCODER_HIFT)


def load_torch_model(config: ExportConfig) -> S3Gen:
    model = S3Gen(meanflow=True)
    state = load_file(config.checkpoint_dir / "s3gen_meanflow.safetensors")
    state = convert_diffusers_transformer_keys(state)
    model.load_state_dict(state, strict=False)
    model.to(config.device).eval()
    model.mel2wav.optimize_for_inference()
    return model


def export(config: ExportConfig) -> None:
    model = load_torch_model(config)
    session = ExportSession(opset=config.opset, external_data=config.external_data)
    out_dir = config.precision_dir
    artifacts = []

    if config.profile in (
        "vc_full_tensor",
        "vc_bucketed",
    ):
        artifacts.append(
            session.export(
                TokenToMuExport(model.flow).to(config.device),
                out_dir / TOKEN_TO_MU,
                tuple(x.to(config.device) for x in token_dummy()),
                token_inputs,
                token_outputs,
                {} if config.profile == "vc_bucketed" else token_axes,
            )
        )
        artifacts.append(
            session.export(
                ConditionalDecoderStepExport(model.flow.decoder.estimator).to(
                    config.device
                ),
                out_dir / CONDITIONAL_DECODER_STEP,
                tuple(x.to(config.device) for x in decoder_dummy()),
                decoder_inputs,
                decoder_outputs,
                {} if config.profile == "vc_bucketed" else decoder_axes,
            )
        )
        artifacts.append(
            session.export(
                FlowDecoderMeanflow2Export(model.flow.decoder).to(config.device),
                out_dir / FLOW_DECODER_MEANFLOW2,
                tuple(x.to(config.device) for x in flow_dummy()),
                flow_inputs,
                flow_outputs,
                {} if config.profile == "vc_bucketed" else flow_axes,
            )
        )
        artifacts.append(
            session.export(
                VocoderExport(model.mel2wav).to(config.device),
                out_dir / VOCODER_HIFT,
                tuple(
                    x.to(config.device)
                    for x in vocoder_dummy(source_hop=model.mel2wav.source_hop)
                ),
                vocoder_inputs,
                vocoder_outputs,
                {} if config.profile == "vc_bucketed" else vocoder_axes,
            )
        )

    write_manifest(config.output_dir, config, artifacts)


def validate_artifacts(artifacts: Path, checkpoint_dir: Path) -> None:
    from .runtime.sessions import OnnxSessions

    OnnxSessions.from_dir(artifacts)
    print(f"Validated loadability for ONNX artifacts under {artifacts}")


def parse_args():
    parser = argparse.ArgumentParser(prog="python -m chatterbox.onnx_export")
    sub = parser.add_subparsers(dest="command", required=True)

    export_p = sub.add_parser("export")
    export_p.add_argument("--checkpoint-dir", required=True, type=Path)
    export_p.add_argument("--output-dir", required=True, type=Path)
    export_p.add_argument("--profile", default="vc_full_tensor")
    export_p.add_argument("--precision", default="fp32")
    export_p.add_argument("--opset", default=18, type=int)
    export_p.add_argument("--device", default="cpu")
    export_p.add_argument("--validate", action="store_true")

    val_p = sub.add_parser("validate")
    val_p.add_argument("--artifacts", required=True, type=Path)
    val_p.add_argument("--checkpoint-dir", required=True, type=Path)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "export":
        config = ExportConfig(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            profile=args.profile,
            precision=args.precision,
            opset=args.opset,
            validate=args.validate,
            device=args.device,
        )
        export(config)
    elif args.command == "validate":
        validate_artifacts(args.artifacts, args.checkpoint_dir)


if __name__ == "__main__":
    main()
