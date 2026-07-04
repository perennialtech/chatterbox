from __future__ import annotations

from pathlib import Path

from ..onnx_export.artifacts import load_manifest, sha256_file
from .config import TrtBuildConfig
from .errors import TensorRTBuildError
from .manifest import EngineRecord, write_trt_manifest
from .shapes import load_shape_plan


def _parser_errors(parser) -> str:
    return "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))


def build_engines(config: TrtBuildConfig) -> list[EngineRecord]:
    import tensorrt as trt

    artifact_dir = Path(config.artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    manifest = load_manifest(artifact_dir)
    shape_plan = load_shape_plan(config.shape_plan)
    output_dir = config.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    records: list[EngineRecord] = []

    for graph_name, graph in manifest["graphs"].items():
        source_rel = graph["files"].get(config.onnx_precision)
        if not source_rel:
            raise TensorRTBuildError(
                f"Graph {graph_name} has no {config.onnx_precision} ONNX artifact"
            )
        source_onnx = artifact_dir / source_rel
        engine_path = output_dir / f"{graph_name}.engine"

        builder = trt.Builder(logger)
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)

        if not parser.parse(source_onnx.read_bytes()):
            raise TensorRTBuildError(
                f"Failed to parse {source_onnx}:\n{_parser_errors(parser)}"
            )

        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, int(config.workspace_bytes)
        )
        if config.engine_precision == "fp16":
            builder_config.set_flag(trt.BuilderFlag.FP16)
        if config.strict_types and hasattr(trt.BuilderFlag, "STRICT_TYPES"):
            builder_config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        profile = builder.create_optimization_profile()
        graph_shapes = shape_plan.get(graph_name, {})
        for i in range(network.num_inputs):
            tensor = network.get_input(i)
            name = tensor.name
            shape = tuple(int(dim) for dim in tensor.shape)
            if any(dim < 0 for dim in shape):
                if name not in graph_shapes:
                    raise TensorRTBuildError(
                        f"Missing TensorRT shape range for dynamic input {graph_name}.{name}"
                    )
                sr = graph_shapes[name]
                profile.set_shape(name, sr.min, sr.opt, sr.max)
        builder_config.add_optimization_profile(profile)

        serialized = builder.build_serialized_network(network, builder_config)
        if serialized is None:
            raise TensorRTBuildError(
                f"TensorRT failed to build engine for {graph_name}"
            )
        engine_path.write_bytes(bytes(serialized))

        records.append(
            EngineRecord(
                graph_name=graph_name,
                engine=engine_path.name,
                source_onnx=str(source_onnx.relative_to(output_dir)),
                source_onnx_hash=sha256_file(source_onnx),
                inputs=list(graph["inputs"]),
                outputs=list(graph["outputs"]),
                shape_ranges=graph_shapes,
            )
        )

    write_trt_manifest(
        output_dir,
        source_manifest=manifest_path,
        precision=config.engine_precision,
        records=records,
        constants=manifest["constants"],
    )
    return records
