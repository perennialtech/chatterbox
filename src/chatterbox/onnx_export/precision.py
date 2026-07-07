from __future__ import annotations

import inspect
import warnings
from pathlib import Path
from typing import Any

from .errors import OnnxExportError


def _iter_graphs(graph: Any):
    yield graph

    try:
        import onnx

        graph_type = onnx.AttributeProto.GRAPH
        graphs_type = getattr(onnx.AttributeProto, "GRAPHS", None)
    except ImportError:
        return

    for node in graph.node:
        for attr in node.attribute:
            if attr.type == graph_type:
                yield from _iter_graphs(attr.g)
            elif graphs_type is not None and attr.type == graphs_type:
                for subgraph in attr.graphs:
                    yield from _iter_graphs(subgraph)


def _clear_intermediate_value_info(model) -> None:
    """Remove intermediate type annotations that can become stale after rewrites."""

    for graph in _iter_graphs(model.graph):
        del graph.value_info[:]


def _collect_value_names(graph) -> set[str]:
    names: set[str] = set()

    def add(name: str) -> None:
        if name:
            names.add(name)

    for value in graph.input:
        add(value.name)
    for value in graph.output:
        add(value.name)
    for value in graph.value_info:
        add(value.name)
    for initializer in graph.initializer:
        add(initializer.name)

    for node in graph.node:
        for name in node.input:
            add(name)
        for name in node.output:
            add(name)

    return names


def _collect_node_names(graph) -> set[str]:
    return {node.name for node in graph.node if node.name}


def _unique_name(used: set[str], base: str) -> str:
    if base not in used:
        used.add(base)
        return base

    index = 1
    while True:
        candidate = f"{base}_{index}"
        if candidate not in used:
            used.add(candidate)
            return candidate
        index += 1


def _tensor_elem_type(value_info) -> int | None:
    if not value_info.type.HasField("tensor_type"):
        return None

    elem_type = value_info.type.tensor_type.elem_type
    return elem_type or None


def _set_tensor_elem_type(value_info, elem_type: int) -> None:
    if value_info.type.HasField("tensor_type"):
        value_info.type.tensor_type.elem_type = elem_type


def _replace_node_inputs(graph, old_name: str, new_name: str) -> int:
    replacements = 0
    for node in graph.node:
        for index, name in enumerate(node.input):
            if name == old_name:
                node.input[index] = new_name
                replacements += 1
    return replacements


def _rename_node_outputs(graph, old_name: str, new_name: str) -> int:
    replacements = 0
    for node in graph.node:
        for index, name in enumerate(node.output):
            if name == old_name:
                node.output[index] = new_name
                replacements += 1
    return replacements


def _prepend_nodes(graph, nodes) -> None:
    if not nodes:
        return

    existing = list(graph.node)
    del graph.node[:]
    graph.node.extend(nodes)
    graph.node.extend(existing)


def _restore_fp32_float_io(onnx, model) -> None:
    """
    Keep exported FP16 artifacts easy to feed from the runtime by exposing float
    graph inputs/outputs as FP32 while running the converted graph body in FP16.

    onnxconverter-common's keep_io_types path can leave mixed FP32/FP16 inputs on
    nodes such as Mul. Converting the full graph first and adding explicit Cast
    nodes at the boundary produces a normal, type-consistent ONNX graph.
    """

    TensorProto = onnx.TensorProto
    helper = onnx.helper
    graph = model.graph

    used_values = _collect_value_names(graph)
    used_nodes = _collect_node_names(graph)

    input_casts = []
    for graph_input in graph.input:
        if _tensor_elem_type(graph_input) != TensorProto.FLOAT16:
            continue

        external_name = graph_input.name
        internal_name = _unique_name(used_values, f"{external_name}__fp16")
        replacements = _replace_node_inputs(graph, external_name, internal_name)

        _set_tensor_elem_type(graph_input, TensorProto.FLOAT)

        if replacements == 0:
            continue

        input_casts.append(
            helper.make_node(
                "Cast",
                [external_name],
                [internal_name],
                name=_unique_name(used_nodes, f"{external_name}__cast_to_fp16"),
                to=TensorProto.FLOAT16,
            )
        )

    _prepend_nodes(graph, input_casts)

    output_casts = []
    for graph_output in graph.output:
        if _tensor_elem_type(graph_output) != TensorProto.FLOAT16:
            continue

        external_name = graph_output.name
        internal_name = _unique_name(used_values, f"{external_name}__fp16")
        producer_count = _rename_node_outputs(graph, external_name, internal_name)

        if producer_count == 0:
            continue

        _replace_node_inputs(graph, external_name, internal_name)
        _set_tensor_elem_type(graph_output, TensorProto.FLOAT)

        output_casts.append(
            helper.make_node(
                "Cast",
                [internal_name],
                [external_name],
                name=_unique_name(used_nodes, f"{external_name}__cast_to_fp32"),
                to=TensorProto.FLOAT,
            )
        )

    graph.node.extend(output_casts)


def _convert_float_to_float16(float16, model):
    kwargs: dict[str, object] = {"keep_io_types": False}
    parameters = inspect.signature(float16.convert_float_to_float16).parameters

    if "disable_shape_infer" in parameters:
        kwargs["disable_shape_infer"] = False

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*truncated.*")
        return float16.convert_float_to_float16(model, **kwargs)


def _infer_shapes_if_possible(onnx, model):
    try:
        return onnx.shape_inference.infer_shapes(
            model, strict_mode=False, data_prop=False
        )
    except TypeError:
        try:
            return onnx.shape_inference.infer_shapes(model)
        except Exception:
            return model
    except Exception:
        return model


def _model_opset_versions(model) -> dict[str, int]:
    return {opset.domain or "": opset.version for opset in model.opset_import}


def _schema_for_node(onnx, node, opset_versions: dict[str, int]):
    domain = node.domain or ""
    version = opset_versions.get(domain)

    try:
        if version is None:
            return onnx.defs.get_schema(node.op_type, domain=domain)
        return onnx.defs.get_schema(
            node.op_type, max_inclusive_version=version, domain=domain
        )
    except Exception:
        return None


def _is_variadic_formal(onnx, formal) -> bool:
    try:
        return formal.option == onnx.defs.OpSchema.FormalParameterOption.Variadic
    except Exception:
        return str(getattr(formal, "option", "")).lower().endswith("variadic")


def _formal_type_strings(onnx, formals, actual_count: int) -> list[str | None]:
    formal_list = list(formals)
    if not formal_list:
        return [None] * actual_count

    result: list[str | None] = []
    formal_index = 0

    while len(result) < actual_count:
        if formal_index >= len(formal_list):
            result.append(None)
            continue

        formal = formal_list[formal_index]
        result.append(formal.type_str or None)

        if not _is_variadic_formal(onnx, formal):
            formal_index += 1
            continue

        if formal_index < len(formal_list) - 1:
            remaining_actual = actual_count - len(result)
            remaining_formals = len(formal_list) - formal_index - 1
            if remaining_actual <= remaining_formals:
                formal_index += 1

    return result


def _type_constraints(schema) -> dict[str, set[str]]:
    return {
        constraint.type_param_str: set(constraint.allowed_type_strs)
        for constraint in schema.type_constraints
    }


def _attr_int(node, name: str) -> int | None:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return None


def _constant_output_elem_type(onnx, node) -> int | None:
    TensorProto = onnx.TensorProto

    for attr in node.attribute:
        if attr.name == "value" and attr.HasField("t"):
            return int(attr.t.data_type) or None
        if attr.name == "sparse_value" and attr.HasField("sparse_tensor"):
            return int(attr.sparse_tensor.values.data_type) or None
        if attr.name in {"value_float", "value_floats"}:
            return TensorProto.FLOAT
        if attr.name in {"value_int", "value_ints"}:
            return TensorProto.INT64
        if attr.name in {"value_string", "value_strings"}:
            return TensorProto.STRING

    return None


def _seed_value_types(onnx, graph) -> dict[str, int]:
    value_types: dict[str, int] = {}

    def add(name: str, elem_type: int | None) -> None:
        if name and elem_type is not None:
            value_types[name] = elem_type

    for value_info in graph.value_info:
        add(value_info.name, _tensor_elem_type(value_info))
    for initializer in graph.initializer:
        add(initializer.name, int(initializer.data_type) or None)
    for value_info in graph.output:
        add(value_info.name, _tensor_elem_type(value_info))
    for value_info in graph.input:
        add(value_info.name, _tensor_elem_type(value_info))

    for node in graph.node:
        if node.op_type == "Constant":
            elem_type = _constant_output_elem_type(onnx, node)
            for output_name in node.output:
                add(output_name, elem_type)

    return value_types


def _elem_type_suffix(onnx, elem_type: int) -> str:
    TensorProto = onnx.TensorProto
    if elem_type == TensorProto.FLOAT16:
        return "fp16"
    if elem_type == TensorProto.FLOAT:
        return "fp32"
    return f"type{elem_type}"


def _node_binding_target_type(
    onnx,
    node,
    type_var: str,
    seen_types: set[int],
    constraints: dict[str, set[str]],
    blocked_ops: set[str],
) -> int | None:
    TensorProto = onnx.TensorProto
    allowed = constraints.get(type_var, set())
    supports_fp32 = "tensor(float)" in allowed
    supports_fp16 = "tensor(float16)" in allowed

    if node.op_type in blocked_ops:
        return TensorProto.FLOAT if supports_fp32 else None

    if not supports_fp16:
        return TensorProto.FLOAT if supports_fp32 else None

    if TensorProto.FLOAT16 in seen_types:
        return TensorProto.FLOAT16

    return None


def _desired_float_casts_for_node(
    onnx,
    node,
    schema,
    input_type_strings: list[str | None],
    constraints: dict[str, set[str]],
    value_types: dict[str, int],
    blocked_ops: set[str],
) -> dict[int, int]:
    TensorProto = onnx.TensorProto
    float_types = {TensorProto.FLOAT, TensorProto.FLOAT16}
    desired_casts: dict[int, int] = {}

    def request(index: int, target_type: int) -> None:
        existing = desired_casts.get(index)
        if existing is None:
            desired_casts[index] = target_type
        elif existing != target_type:
            desired_casts[index] = TensorProto.FLOAT

    for index, (input_name, type_string) in enumerate(
        zip(node.input, input_type_strings)
    ):
        if not input_name or type_string is None:
            continue

        elem_type = value_types.get(input_name)
        if elem_type not in float_types:
            continue

        if type_string == "tensor(float)" and elem_type != TensorProto.FLOAT:
            request(index, TensorProto.FLOAT)
        elif type_string == "tensor(float16)" and elem_type != TensorProto.FLOAT16:
            request(index, TensorProto.FLOAT16)

    grouped: dict[str, list[tuple[int, str, int]]] = {}
    for index, (input_name, type_var) in enumerate(zip(node.input, input_type_strings)):
        if not input_name or type_var not in constraints:
            continue

        elem_type = value_types.get(input_name)
        if elem_type in float_types:
            grouped.setdefault(type_var, []).append((index, input_name, elem_type))

    for type_var, entries in grouped.items():
        seen_types = {elem_type for _, _, elem_type in entries}
        allowed = constraints.get(type_var, set())
        supports_fp16 = "tensor(float16)" in allowed
        must_rebind = (
            len(seen_types) > 1
            or (TensorProto.FLOAT16 in seen_types and not supports_fp16)
            or (TensorProto.FLOAT16 in seen_types and node.op_type in blocked_ops)
        )

        if not must_rebind:
            continue

        target_type = _node_binding_target_type(
            onnx, node, type_var, seen_types, constraints, blocked_ops
        )
        if target_type is None:
            continue

        for index, _, elem_type in entries:
            if elem_type != target_type:
                request(index, target_type)

    return desired_casts


def _insert_requested_casts(
    onnx,
    graph,
    node,
    desired_casts: dict[int, int],
    value_types: dict[str, int],
    used_values: set[str],
    used_nodes: set[str],
) -> list[Any]:
    helper = onnx.helper
    cast_nodes = []
    local_cache: dict[tuple[str, int], str] = {}

    for index in sorted(desired_casts):
        input_name = node.input[index]
        target_type = desired_casts[index]

        if not input_name or value_types.get(input_name) == target_type:
            continue

        cache_key = (input_name, target_type)
        cast_output = local_cache.get(cache_key)

        if cast_output is None:
            suffix = _elem_type_suffix(onnx, target_type)
            cast_output = _unique_name(used_values, f"{input_name}__cast_to_{suffix}")
            cast_nodes.append(
                helper.make_node(
                    "Cast",
                    [input_name],
                    [cast_output],
                    name=_unique_name(
                        used_nodes,
                        f"{node.name or node.op_type}_input_{index}__cast_to_{suffix}",
                    ),
                    to=target_type,
                )
            )
            local_cache[cache_key] = cast_output
            value_types[cast_output] = target_type

        node.input[index] = cast_output

    return cast_nodes


def _update_node_output_types(
    onnx,
    node,
    schema,
    input_type_strings: list[str | None],
    constraints: dict[str, set[str]],
    value_types: dict[str, int],
) -> None:
    if node.op_type == "Constant":
        elem_type = _constant_output_elem_type(onnx, node)
        if elem_type is not None:
            for output_name in node.output:
                if output_name:
                    value_types[output_name] = elem_type
        return

    if node.op_type == "Cast":
        elem_type = _attr_int(node, "to")
        if elem_type is not None:
            for output_name in node.output:
                if output_name:
                    value_types[output_name] = elem_type
        return

    if schema is None:
        return

    bindings: dict[str, int] = {}
    for input_name, type_var in zip(node.input, input_type_strings):
        if not input_name or type_var not in constraints:
            continue

        elem_type = value_types.get(input_name)
        if elem_type is not None:
            bindings.setdefault(type_var, elem_type)

    output_type_strings = _formal_type_strings(onnx, schema.outputs, len(node.output))
    for output_name, type_string in zip(node.output, output_type_strings):
        if not output_name or type_string is None:
            continue

        if type_string in bindings:
            value_types[output_name] = bindings[type_string]
        elif type_string == "tensor(float)":
            value_types[output_name] = onnx.TensorProto.FLOAT
        elif type_string == "tensor(float16)":
            value_types[output_name] = onnx.TensorProto.FLOAT16


def _repair_graph_mixed_float_type_bindings(
    onnx,
    graph,
    opset_versions: dict[str, int],
    blocked_ops: set[str],
) -> int:
    used_values = _collect_value_names(graph)
    used_nodes = _collect_node_names(graph)
    value_types = _seed_value_types(onnx, graph)

    changed = 0
    rewritten_nodes = []

    for node in list(graph.node):
        schema = _schema_for_node(onnx, node, opset_versions)
        constraints = _type_constraints(schema) if schema is not None else {}

        if schema is not None:
            input_type_strings = _formal_type_strings(
                onnx, schema.inputs, len(node.input)
            )
            desired_casts = _desired_float_casts_for_node(
                onnx,
                node,
                schema,
                input_type_strings,
                constraints,
                value_types,
                blocked_ops,
            )
        else:
            input_type_strings = [None] * len(node.input)
            desired_casts = {}

        if desired_casts:
            cast_nodes = _insert_requested_casts(
                onnx,
                graph,
                node,
                desired_casts,
                value_types,
                used_values,
                used_nodes,
            )
            changed += len(cast_nodes)
            rewritten_nodes.extend(cast_nodes)

        rewritten_nodes.append(node)
        _update_node_output_types(
            onnx, node, schema, input_type_strings, constraints, value_types
        )

    if changed:
        del graph.node[:]
        graph.node.extend(rewritten_nodes)

    return changed


def _repair_mixed_float_type_bindings(onnx, model, blocked_ops: set[str]) -> int:
    opset_versions = _model_opset_versions(model)
    repaired = 0

    for graph in _iter_graphs(model.graph):
        repaired += _repair_graph_mixed_float_type_bindings(
            onnx, graph, opset_versions, blocked_ops
        )

    return repaired


def _remove_existing_output(path: Path) -> None:
    if path.exists():
        path.unlink()

    data_path = path.with_name(path.name + ".data")
    if data_path.exists():
        data_path.unlink()


def _save_model(onnx, model, path: Path, external_data: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _remove_existing_output(path)

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

    try:
        model = onnx.load(str(src), load_external_data=True)
        _clear_intermediate_value_info(model)

        model_fp16 = _convert_float_to_float16(float16, model)
        model_fp16 = _infer_shapes_if_possible(onnx, model_fp16)

        _restore_fp32_float_io(onnx, model_fp16)
        model_fp16 = _infer_shapes_if_possible(onnx, model_fp16)

        blocked_ops = set(getattr(float16, "DEFAULT_OP_BLOCK_LIST", ()))
        _repair_mixed_float_type_bindings(onnx, model_fp16, blocked_ops)
        model_fp16 = _infer_shapes_if_possible(onnx, model_fp16)

        _clear_intermediate_value_info(model_fp16)

        _save_model(onnx, model_fp16, dst, external_data=external_data)
        onnx.checker.check_model(str(dst))
        _check_ort_load(dst)
    except OnnxExportError:
        raise
    except Exception as exc:
        raise OnnxExportError(f"Failed to convert ONNX model to FP16: {src}") from exc
