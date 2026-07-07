from __future__ import annotations

from pathlib import Path

from ..onnx_export.artifacts import load_manifest, sha256_file
from .api import network_creation_flags, require_tensorrt_10
from .config import TrtBuildConfig
from .cuda import cuda_memory_info
from .errors import TensorRTBuildError
from .manifest import EngineRecord, write_trt_manifest
from .shapes import load_shape_plan
from .types import ShapeRange


def _parser_errors(parser) -> str:
    return "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))


def _validate_config(config: TrtBuildConfig) -> None:
    if config.workspace_bytes <= 0:
        raise TensorRTBuildError("TensorRT workspace size must be positive")


def _configure_precision(
    trt, builder_config, config: TrtBuildConfig, precision: str
) -> None:
    if hasattr(trt.BuilderFlag, "TF32"):
        builder_config.clear_flag(trt.BuilderFlag.TF32)

    if config.strongly_typed:
        return

    if precision == "fp16":
        if hasattr(trt.BuilderFlag, "FP16"):
            builder_config.set_flag(trt.BuilderFlag.FP16)


def _force_linear_io_formats(trt, network) -> None:
    tensor_format = getattr(trt, "TensorFormat", None)
    linear_format = getattr(tensor_format, "LINEAR", None) if tensor_format else None
    if linear_format is None:
        raise TensorRTBuildError("TensorRT LINEAR tensor format is not available")

    linear_mask = 1 << int(linear_format)

    for i in range(network.num_inputs):
        network.get_input(i).allowed_formats = linear_mask

    for i in range(network.num_outputs):
        network.get_output(i).allowed_formats = linear_mask


def _effective_workspace_bytes(requested_bytes: int) -> int:
    try:
        free_bytes, _ = cuda_memory_info()
    except Exception:
        return requested_bytes

    reserve_bytes = max(512 * 1024**2, free_bytes // 4)
    cap_bytes = max(1 * 1024**2, free_bytes - reserve_bytes)
    return min(requested_bytes, cap_bytes)


def _validate_profile_shape(
    graph_name: str,
    input_name: str,
    network_shape: tuple[int, ...],
    shape_range: ShapeRange,
) -> None:
    if not (
        len(network_shape)
        == len(shape_range.min)
        == len(shape_range.opt)
        == len(shape_range.max)
    ):
        raise TensorRTBuildError(
            f"{graph_name}.{input_name}: TensorRT profile rank does not match ONNX input rank"
        )

    for axis, (declared, mn, opt, mx) in enumerate(
        zip(network_shape, shape_range.min, shape_range.opt, shape_range.max)
    ):
        if declared >= 0 and (mn, opt, mx) != (declared, declared, declared):
            raise TensorRTBuildError(
                f"{graph_name}.{input_name}: profile axis {axis} must match "
                f"static ONNX dimension {declared}; got min/opt/max={mn}/{opt}/{mx}"
            )


def _add_optimization_profile(
    trt,
    builder,
    builder_config,
    network,
    graph_name: str,
    graph_shapes: dict[str, ShapeRange],
) -> None:
    profile = builder.create_optimization_profile()
    uses_profile = False

    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        input_name = tensor.name
        network_shape = tuple(int(dim) for dim in tensor.shape)

        if not any(dim < 0 for dim in network_shape):
            continue

        if input_name not in graph_shapes:
            raise TensorRTBuildError(
                f"Missing TensorRT shape range for dynamic input {graph_name}.{input_name}"
            )

        shape_range = graph_shapes[input_name]
        _validate_profile_shape(graph_name, input_name, network_shape, shape_range)
        profile.set_shape(input_name, shape_range.min, shape_range.opt, shape_range.max)
        uses_profile = True

    if not uses_profile:
        return

    profile_index = builder_config.add_optimization_profile(profile)
    if profile_index < 0:
        raise TensorRTBuildError(
            f"TensorRT rejected optimization profile for {graph_name}"
        )


def build_engines(config: TrtBuildConfig) -> list[EngineRecord]:
    import tensorrt as trt

    require_tensorrt_10(trt, TensorRTBuildError)
    _validate_config(config)

    artifact_dir = Path(config.artifact_dir)
    manifest_path = artifact_dir / "manifest.json"
    manifest = load_manifest(artifact_dir)
    precision = manifest.get("precision", "fp32")
    source_hop = int(manifest.get("constants", {}).get("source_hop", 480))
    shape_plan = load_shape_plan(config.shape_plan, source_hop=source_hop)
    output_dir = config.resolved_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = trt.Logger(trt.Logger.WARNING)
    records: list[EngineRecord] = []

    for graph_name, graph in manifest["graphs"].items():
        source_rel = graph["files"].get(precision)
        if not source_rel:
            raise TensorRTBuildError(
                f"Graph {graph_name} has no {precision} ONNX artifact"
            )

        source_rel_path = Path(source_rel)
        source_onnx = artifact_dir / source_rel_path
        if not source_onnx.exists():
            raise TensorRTBuildError(
                f"Missing ONNX artifact for {graph_name}: {source_onnx}"
            )

        engine_path = output_dir / f"{graph_name}.engine"

        builder = trt.Builder(logger)
        flags = network_creation_flags(trt, strongly_typed=config.strongly_typed)
        network = builder.create_network(flags)
        parser = trt.OnnxParser(network, logger)

        if not parser.parse_from_file(str(source_onnx)):
            raise TensorRTBuildError(
                f"Failed to parse {source_onnx}:\n{_parser_errors(parser)}"
            )

        _force_linear_io_formats(trt, network)

        builder_config = builder.create_builder_config()
        builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE,
            int(_effective_workspace_bytes(config.workspace_bytes)),
        )
        _configure_precision(trt, builder_config, config, precision)

        graph_shapes = shape_plan.get(graph_name, {})
        _add_optimization_profile(
            trt,
            builder,
            builder_config,
            network,
            graph_name,
            graph_shapes,
        )

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
                source_onnx=source_rel_path.as_posix(),
                source_onnx_hash=sha256_file(source_onnx),
                inputs=list(graph["inputs"]),
                outputs=list(graph["outputs"]),
                shape_ranges=graph_shapes,
            )
        )

    write_trt_manifest(
        output_dir,
        source_manifest=manifest_path,
        precision=precision,
        records=records,
        constants=manifest["constants"],
    )
    return records
