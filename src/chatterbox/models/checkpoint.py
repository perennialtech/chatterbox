from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from chatterbox.device import Runtime
from chatterbox.types import ReferenceConditioning

if TYPE_CHECKING:
    from chatterbox.models.pipeline import VoiceConversionPipeline

REPO_ID = "ResembleAI/chatterbox-turbo"


@dataclass(frozen=True)
class CheckpointBundle:
    root: Path
    model_weights: Path
    config: Path
    builtin_reference: Path | None


@dataclass(frozen=True)
class PipelineConfig:
    tokenizer_sr: int = 16_000
    sample_rate: int = 24_000
    max_reference_seconds: int = 10
    model_revision: int = 2


class CheckpointLoader:
    @staticmethod
    def from_local(path: str | Path) -> CheckpointBundle:
        root = Path(path).expanduser().resolve()
        model_weights = root / "model.safetensors"
        config = root / "config.json"
        builtin_reference = root / "builtin_reference.safetensors"

        if not model_weights.exists():
            raise FileNotFoundError(model_weights)
        if not config.exists():
            raise FileNotFoundError(config)

        return CheckpointBundle(
            root=root,
            model_weights=model_weights,
            config=config,
            builtin_reference=builtin_reference if builtin_reference.exists() else None,
        )

    @staticmethod
    def from_pretrained(repo_id: str = REPO_ID) -> CheckpointBundle:
        root = Path(
            snapshot_download(
                repo_id=repo_id,
                allow_patterns=[
                    "model.safetensors",
                    "config.json",
                    "builtin_reference.safetensors",
                ],
            )
        )
        return CheckpointLoader.from_local(root)

    @staticmethod
    def load_pipeline(
        bundle: CheckpointBundle, runtime: Runtime
    ) -> "VoiceConversionPipeline":
        from chatterbox.models.pipeline import VoiceConversionPipeline

        with bundle.config.open("r", encoding="utf-8") as f:
            config = PipelineConfig(**json.load(f))

        pipeline = VoiceConversionPipeline(config=config, runtime=runtime)
        state = load_file(bundle.model_weights, device="cpu")
        pipeline.load_state_dict(state, strict=True)
        pipeline.to(runtime.device).eval()

        if runtime.compile:
            pipeline.compile_for_inference()

        return pipeline

    @staticmethod
    def load_builtin_reference(
        bundle: CheckpointBundle, runtime: Runtime
    ) -> ReferenceConditioning | None:
        if bundle.builtin_reference is None:
            return None

        state = load_file(bundle.builtin_reference, device=str(runtime.device))
        return ReferenceConditioning(
            prompt_tokens=state["prompt_tokens"].long(),
            prompt_token_lengths=state["prompt_token_lengths"].long(),
            prompt_mels=state["prompt_mels"].to(dtype=runtime.compute_dtype),
            prompt_mel_lengths=state["prompt_mel_lengths"].long(),
            speaker_embedding=state["speaker_embedding"].to(
                dtype=runtime.compute_dtype
            ),
        )

    @staticmethod
    def write_config(path: str | Path, config: PipelineConfig) -> None:
        path = Path(path)
        path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
