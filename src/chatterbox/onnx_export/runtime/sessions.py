from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..artifacts import load_manifest
from ..errors import OnnxRuntimeError
from .runner import OnnxGraphRunner


@dataclass
class OnnxSessions:
    artifact_dir: Path
    manifest: dict
    sessions: dict[str, object]

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: Path,
        providers: list[str] | None = None,
    ) -> "OnnxSessions":
        import onnxruntime as ort

        artifact_dir = Path(artifact_dir).resolve()
        manifest = load_manifest(artifact_dir)

        if providers is None:
            providers = ["CPUExecutionProvider"]
        else:
            available = set(ort.get_available_providers())
            providers = [provider for provider in providers if provider in available]
            if not providers:
                raise OnnxRuntimeError(
                    "None of the requested ONNX providers is available"
                )

        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        sessions = {}
        for graph_name, graph in manifest["graphs"].items():
            if graph.get("required_for_runtime", False):
                file_rel = graph.get("path")
                if not file_rel:
                    raise OnnxRuntimeError(
                        f"Required graph {graph_name} has no ONNX artifact path"
                    )
                path = artifact_dir / file_rel
                if not path.exists():
                    raise OnnxRuntimeError(
                        f"Missing ONNX artifact for {graph_name}: {path}"
                    )
                sessions[graph_name] = ort.InferenceSession(
                    str(path),
                    sess_options=session_options,
                    providers=providers,
                )

        return cls(
            artifact_dir=artifact_dir,
            manifest=manifest,
            sessions=sessions,
        )

    @classmethod
    def from_dir(
        cls, artifact_dir: Path, providers: list[str] | None = None
    ) -> "OnnxSessions":
        return cls.from_artifact_dir(artifact_dir, providers=providers)

    def require(self, graph_name: str):
        if graph_name not in self.sessions:
            raise OnnxRuntimeError(f"Required ONNX graph is not loaded: {graph_name}")
        return self.sessions[graph_name]

    def runner(self, graph_name: str) -> OnnxGraphRunner:
        graph = self.manifest["graphs"][graph_name]
        session = self.require(graph_name)
        return OnnxGraphRunner(
            name=graph_name,
            session=session,
            input_names=list(graph["inputs"]),
            output_names=list(graph["outputs"]),
            actual_input_names=[inp.name for inp in session.get_inputs()],
        )
