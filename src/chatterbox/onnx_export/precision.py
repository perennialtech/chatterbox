from __future__ import annotations

from pathlib import Path

from .errors import OnnxExportError


def _clear_intermediate_value_info(model) -> None:
    """Remove stale intermediate type annotations before/after FP16 rewriting."""

    import onnx

    graph_type = onnx.AttributeProto.GRAPH
    graphs_type = getattr(onnx.AttributeProto, "GRAPHS", None)

    def clear_graph(graph) -> None:
        del graph.value_info[:]

        for node in graph.node:
            for attr in node.attribute:
                if attr.type == graph_type:
                    clear_graph(attr.g)
                elif graphs_type is not None and attr.type == graphs_type:
                    for subgraph in attr.graphs:
                        clear_graph(subgraph)

    clear_graph(model.graph)


def _convert_float_to_float16(float16, model):
    try:
        return float16.convert_float_to_float16(
            model,
            keep_io_types=True,
            disable_shape_infer=True,
        )
    except TypeError:
        return float16.convert_float_to_float16(model, keep_io_types=True)


def _remove_existing_output(path: Path, external_data: bool) -> None:
    if path.exists():
        path.unlink()

    if external_data:
        data_path = path.with_name(path.name + ".data")
        if data_path.exists():
            data_path.unlink()


def _save_model(onnx, model, path: Path, external_data: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_output(path, external_data)

    if external_data:
        onnx.save_model(
            model,
            str(path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=path.name + ".data",
            size_threshold=1024,
            convert_attribute=False,
        )
    else:
        onnx.save(model, str(path))


def _check_ort_load(path: Path) -> None:
    try:
        import onnxruntime as ort
    except ImportError:
        return

    try:
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        ort.InferenceSession(
            str(path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        raise OnnxExportError(f"ONNX Runtime failed to load FP16 model {path}") from exc


def convert_fp16(src: Path, dst: Path, external_data: bool = True) -> None:
    try:
        import onnx
        from onnxconverter_common import float16
    except ImportError as exc:
        raise OnnxExportError(
            "onnx and onnxconverter-common are required for FP16 conversion"
        ) from exc

    model = onnx.load(str(src), load_external_data=True)
    _clear_intermediate_value_info(model)

    model_fp16 = _convert_float_to_float16(float16, model)
    _clear_intermediate_value_info(model_fp16)

    _save_model(onnx, model_fp16, dst, external_data=external_data)
    onnx.checker.check_model(str(dst))
    _check_ort_load(dst)
