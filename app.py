from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Any

import gradio as gr
import numpy as np

APP_TITLE = "Chatterbox Voice Conversion"
BACKEND_CHOICES = ("pytorch",)

LOGGER = logging.getLogger("chatterbox.app")
logging.basicConfig(
    level=os.environ.get("CHATTERBOX_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)


@dataclass(frozen=True)
class AppSettings:
    backend: str
    device: str


@dataclass(frozen=True)
class LaunchSettings:
    server_name: str
    server_port: int
    share: bool
    inbrowser: bool
    debug: bool


@dataclass(frozen=True)
class RuntimeConfig:
    backend: str
    device: str = ""


@dataclass(frozen=True)
class RuntimeResult:
    wav: Any
    sample_rate: int
    timings: Any
    loaded: bool
    load_seconds: float
    generation_seconds: float


class RuntimeManager:
    """Lazily loads and serializes access to the selected Chatterbox backend."""

    def __init__(self, config: RuntimeConfig) -> None:
        self._lock = RLock()
        self._config = config
        self._vc: Any | None = None
        self._load_seconds = 0.0

    def generate(
        self,
        source_path: Path,
        target_voice_path: Path,
    ) -> RuntimeResult:
        with self._lock:
            vc, loaded, load_seconds = self._runtime()
            started = time.perf_counter()
            wav, sample_rate, timings = vc.generate(
                source_path,
                target_voice_path=target_voice_path,
            )
            generation_seconds = time.perf_counter() - started

        return RuntimeResult(
            wav=wav,
            sample_rate=int(sample_rate),
            timings=timings,
            loaded=loaded,
            load_seconds=load_seconds,
            generation_seconds=generation_seconds,
        )

    def _runtime(self) -> tuple[Any, bool, float]:
        if self._vc is not None:
            return self._vc, False, self._load_seconds

        LOGGER.info("Loading Chatterbox runtime: %s", _runtime_label(self._config))
        started = time.perf_counter()
        vc = self._load_runtime(self._config)
        load_seconds = time.perf_counter() - started

        self._vc = vc
        self._load_seconds = load_seconds
        LOGGER.info("Loaded Chatterbox runtime in %.3fs", load_seconds)
        return vc, True, load_seconds

    @staticmethod
    def _load_runtime(config: RuntimeConfig) -> Any:
        try:
            from chatterbox import ChatterboxVC
        except Exception as exc:  # pragma: no cover - import diagnostics
            raise RuntimeError(
                "Unable to import chatterbox. Install the project with the runtime "
                "extras required by the selected backend."
            ) from exc

        if config.backend == "pytorch":
            return ChatterboxVC.from_pretrained(device=config.device)

        raise ValueError(f"Unsupported backend: {config.backend}")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _default_backend() -> str:
    backend = os.environ.get("CHATTERBOX_BACKEND", "pytorch").strip().lower()
    if backend in BACKEND_CHOICES:
        return backend
    return "pytorch"


def _default_device() -> str:
    configured = os.environ.get("CHATTERBOX_DEVICE")
    if configured:
        return configured.strip()

    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def default_app_settings() -> AppSettings:
    return AppSettings(
        backend=_default_backend(),
        device=_default_device(),
    )


def default_launch_settings() -> LaunchSettings:
    return LaunchSettings(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=_env_int("GRADIO_SERVER_PORT", 7860),
        share=_env_bool("GRADIO_SHARE", False),
        inbrowser=_env_bool("GRADIO_INBROWSER", False),
        debug=_env_bool("GRADIO_DEBUG", False),
    )


def parse_args() -> tuple[AppSettings, LaunchSettings]:
    app_defaults = default_app_settings()
    launch_defaults = default_launch_settings()

    parser = argparse.ArgumentParser(description="Launch the Chatterbox Gradio app.")
    parser.add_argument(
        "--backend",
        choices=BACKEND_CHOICES,
        default=app_defaults.backend,
        help="Runtime backend to use.",
    )
    parser.add_argument(
        "--device",
        "--torch-device",
        dest="device",
        default=app_defaults.device,
        help="PyTorch device used by the pytorch backend.",
    )
    parser.add_argument(
        "--server-name",
        default=launch_defaults.server_name,
        help="Host interface passed to gradio.Blocks.launch.",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=launch_defaults.server_port,
        help="Port passed to gradio.Blocks.launch.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        default=launch_defaults.share,
        help="Create a public Gradio share URL.",
    )
    parser.add_argument(
        "--inbrowser",
        action="store_true",
        default=launch_defaults.inbrowser,
        help="Open the app in a browser after launch.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=launch_defaults.debug,
        help="Run Gradio in debug mode.",
    )

    args = parser.parse_args()
    return (
        AppSettings(
            backend=args.backend,
            device=args.device,
        ),
        LaunchSettings(
            server_name=args.server_name,
            server_port=args.server_port,
            share=args.share,
            inbrowser=args.inbrowser,
            debug=args.debug,
        ),
    )


def _audio_path(value: Any, label: str) -> Path:
    if value is None:
        raise gr.Error(f"{label} audio is required.")

    if isinstance(value, (str, Path)):
        path_value = value
    elif isinstance(value, dict):
        path_value = value.get("path") or value.get("name")
    else:
        path_value = None

    if not path_value:
        raise gr.Error(f"{label} audio must be provided as a file path.")

    path = Path(path_value).expanduser()
    if not path.exists():
        raise gr.Error(f"{label} audio file does not exist: {path}")
    if not path.is_file():
        raise gr.Error(f"{label} audio must be a file: {path}")
    return path


def _runtime_config(
    backend: str,
    device: str,
) -> RuntimeConfig:
    selected_backend = str(backend or "").strip().lower()
    if selected_backend not in BACKEND_CHOICES:
        raise ValueError(f"Unsupported backend: {backend}")

    return RuntimeConfig(
        backend="pytorch",
        device=str(device or "cpu").strip() or "cpu",
    )


def _runtime_label(config: RuntimeConfig) -> str:
    if config.backend == "pytorch":
        return f"PyTorch device={config.device}"

    return config.backend


def _runtime_payload(config: RuntimeConfig) -> dict[str, Any]:
    if config.backend == "pytorch":
        return {"backend": "pytorch", "device": config.device}

    return {"backend": config.backend}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "detach"):
        try:
            tensor = value.detach().cpu()
            if tensor.ndim == 0:
                return tensor.item()
            return tensor.numpy().tolist()
        except Exception:
            return str(value)
    return str(value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _format_seconds(value: float) -> str:
    if value < 0.001:
        return f"{value * 1_000_000:.1f} µs"
    if value < 1.0:
        return f"{value * 1_000:.1f} ms"
    return f"{value:.3f} s"


def _to_gradio_audio(wav: Any, sample_rate: int) -> tuple[int, np.ndarray]:
    value = wav
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "float"):
        value = value.float()
    if hasattr(value, "numpy"):
        value = value.numpy()

    audio = np.asarray(value, dtype=np.float32)

    while audio.ndim > 1 and audio.shape[0] == 1:
        audio = audio[0]

    if audio.ndim == 0:
        audio = audio.reshape(1)
    elif audio.ndim == 2:
        # Chatterbox and torchaudio commonly use channel-first tensors.
        # Gradio expects mono [samples] or multichannel [samples, channels].
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            audio = audio.T
    elif audio.ndim > 2:
        audio = np.squeeze(audio)
        if audio.ndim > 2:
            audio = audio.reshape(-1)

    audio = np.nan_to_num(audio, nan=0.0, posinf=1.0, neginf=-1.0)
    audio = np.clip(audio, -1.0, 1.0)
    return int(sample_rate), np.ascontiguousarray(audio)


def _audio_duration_seconds(audio: np.ndarray, sample_rate: int) -> float:
    if sample_rate <= 0 or audio.size == 0:
        return 0.0
    if audio.ndim == 1:
        samples = audio.shape[0]
    else:
        samples = audio.shape[0]
    return samples / sample_rate


def _status_markdown(
    config: RuntimeConfig,
    result: RuntimeResult,
    duration_seconds: float,
    request_seconds: float,
) -> str:
    timings = result.timings if isinstance(result.timings, dict) else {}
    model_rtf = _to_float(timings.get("rtf"))
    wall_rtf = request_seconds / duration_seconds if duration_seconds > 0 else None

    if result.loaded:
        load_line = f"- Runtime load: {_format_seconds(result.load_seconds)}"
    else:
        load_line = f"- Runtime load: cached ({_format_seconds(result.load_seconds)})"

    lines = [
        "### Conversion complete",
        f"- Runtime: `{_runtime_label(config)}`",
        f"- Sample rate: `{result.sample_rate}` Hz",
        f"- Output duration: `{duration_seconds:.3f}` s",
        load_line,
        f"- Generation call: {_format_seconds(result.generation_seconds)}",
        f"- App wall time: {_format_seconds(request_seconds)}",
    ]

    if wall_rtf is not None:
        lines.append(f"- App wall RTF: `{wall_rtf:.3f}`")
    if model_rtf is not None:
        lines.append(f"- Backend timing RTF: `{model_rtf:.3f}`")

    return "\n".join(lines)


def build_demo(settings: AppSettings) -> gr.Blocks:
    config = _runtime_config(
        backend=settings.backend,
        device=settings.device,
    )
    manager = RuntimeManager(config)

    def convert_voice(
        source_audio: Any,
        target_voice_audio: Any,
    ) -> tuple[tuple[int, np.ndarray], str, dict[str, Any]]:
        source_path = _audio_path(source_audio, "Source")
        target_voice_path = _audio_path(target_voice_audio, "Target voice")

        request_started = time.perf_counter()
        try:
            result = manager.generate(source_path, target_voice_path)
        except gr.Error:
            raise
        except Exception as exc:
            LOGGER.exception("Voice conversion failed")
            raise gr.Error(f"Voice conversion failed: {exc}") from exc

        request_seconds = time.perf_counter() - request_started
        output_audio = _to_gradio_audio(result.wav, result.sample_rate)
        duration_seconds = _audio_duration_seconds(output_audio[1], output_audio[0])
        status = _status_markdown(config, result, duration_seconds, request_seconds)

        details = {
            "runtime": _runtime_payload(config),
            "source": str(source_path),
            "target_voice": str(target_voice_path),
            "sample_rate": result.sample_rate,
            "output_duration_seconds": duration_seconds,
            "request_seconds": request_seconds,
            "runtime_loaded_for_request": result.loaded,
            "runtime_load_seconds": result.load_seconds,
            "generation_seconds": result.generation_seconds,
            "model_timings": _jsonable(result.timings),
        }
        return output_audio, status, _jsonable(details)

    with gr.Blocks(title=APP_TITLE) as demo:
        gr.Markdown(f"""
# {APP_TITLE}

Upload source speech and a target-voice reference, and run voice conversion.
The configured runtime is loaded lazily and cached.

Use clean target speech for best voice conditioning.
""")

        with gr.Row():
            source_audio = gr.Audio(
                label="Source speech",
                type="filepath",
            )
            target_voice_audio = gr.Audio(
                label="Target voice reference",
                type="filepath",
            )

        convert_button = gr.Button("Convert voice", variant="primary")

        with gr.Row():
            with gr.Column(scale=1):
                output_audio = gr.Audio(
                    label="Converted voice",
                    type="numpy",
                )
            with gr.Column(scale=1):
                status = gr.Markdown("Load audio and press **Convert voice**.")
                details = gr.JSON(label="Run details")

        convert_button.click(
            fn=convert_voice,
            inputs=[
                source_audio,
                target_voice_audio,
            ],
            outputs=[output_audio, status, details],
            api_name="convert",
        )

    return demo


demo = build_demo(default_app_settings())


def main() -> None:
    app_settings, launch_settings = parse_args()
    local_demo = build_demo(app_settings)
    local_demo.queue()
    local_demo.launch(
        server_name=launch_settings.server_name,
        server_port=launch_settings.server_port,
        share=launch_settings.share,
        inbrowser=launch_settings.inbrowser,
        debug=launch_settings.debug,
        show_error=True,
    )


if __name__ == "__main__":
    main()
