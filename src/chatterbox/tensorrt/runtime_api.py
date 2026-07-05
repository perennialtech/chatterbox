from __future__ import annotations

from typing import Any

from .errors import TensorRTRuntimeError


def engine_device_memory_size_v2(engine) -> int:
    """Return the engine's required execution-context device memory in bytes."""

    for name in (
        "get_device_memory_size_v2",
        "get_device_memory_size",
    ):
        getter = getattr(engine, name, None)
        if getter is None:
            continue
        try:
            nbytes = int(getter())
        except TypeError:
            continue
        if nbytes < 0:
            raise TensorRTRuntimeError(
                f"TensorRT reported a negative device memory size: {nbytes}"
            )
        return nbytes

    for name in (
        "device_memory_size_v2",
        "device_memory_size",
    ):
        if not hasattr(engine, name):
            continue
        nbytes = int(getattr(engine, name))
        if nbytes < 0:
            raise TensorRTRuntimeError(
                f"TensorRT reported a negative device memory size: {nbytes}"
            )
        return nbytes

    for name in (
        "get_device_memory_size_for_profile_v2",
        "get_device_memory_size_for_profile",
    ):
        getter = getattr(engine, name, None)
        if getter is None:
            continue
        try:
            nbytes = int(getter(0))
        except TypeError:
            continue
        if nbytes < 0:
            raise TensorRTRuntimeError(
                f"TensorRT reported a negative device memory size: {nbytes}"
            )
        return nbytes

    raise TensorRTRuntimeError(
        "TensorRT Python bindings do not expose engine device memory sizing"
    )


def create_user_managed_context(trt_module, engine):
    """Create an execution context that expects caller-owned device memory."""

    create_context = getattr(engine, "create_execution_context", None)
    strategy_enum = getattr(trt_module, "ExecutionContextAllocationStrategy", None)

    if create_context is not None and strategy_enum is not None:
        for member_name in ("USER_MANAGED", "kUSER_MANAGED"):
            if not hasattr(strategy_enum, member_name):
                continue

            strategy = getattr(strategy_enum, member_name)
            context = _try_create_context_with_strategy(create_context, strategy)
            if context is not None:
                return context

    create_without_memory = getattr(
        engine, "create_execution_context_without_device_memory", None
    )
    if create_without_memory is not None:
        context = create_without_memory()
        if context is not None:
            return context

    raise TensorRTRuntimeError(
        "TensorRT Python bindings do not expose user-managed execution contexts"
    )


def set_context_device_memory(context, ptr: int, size: int) -> None:
    """Assign caller-owned activation memory to a TensorRT execution context."""

    ptr = int(ptr)
    size = int(size)

    if ptr == 0:
        raise TensorRTRuntimeError(
            "Cannot assign a null TensorRT device memory pointer"
        )
    if size <= 0:
        raise TensorRTRuntimeError(
            f"Cannot assign non-positive TensorRT device memory size: {size}"
        )

    for method_name in (
        "set_device_memory_v2",
        "set_device_memory",
    ):
        method = getattr(context, method_name, None)
        if method is None:
            continue
        if _try_set_device_memory_with_method(method, ptr, size):
            return

    try:
        setattr(context, "device_memory", ptr)
    except Exception as exc:
        raise TensorRTRuntimeError(
            "TensorRT Python bindings do not expose context device memory assignment"
        ) from exc


def _try_create_context_with_strategy(create_context, strategy: Any):
    attempts = (
        lambda: create_context(strategy),
        lambda: create_context(strategy=strategy),
        lambda: create_context(allocation_strategy=strategy),
    )

    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            return attempt()
        except TypeError as exc:
            last_exc = exc

    if last_exc is not None:
        return None

    return None


def _try_set_device_memory_with_method(method, ptr: int, size: int) -> bool:
    attempts = (
        lambda: method(ptr, size),
        lambda: method(memory=ptr, size=size),
        lambda: method(ptr),
        lambda: method(memory=ptr),
    )

    matched_signature = False
    for attempt in attempts:
        try:
            result = attempt()
        except TypeError:
            continue

        matched_signature = True
        if result is False:
            raise TensorRTRuntimeError("TensorRT rejected context device memory")
        return True

    return matched_signature
