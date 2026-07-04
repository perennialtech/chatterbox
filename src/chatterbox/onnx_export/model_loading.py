from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file

from ..models.s3gen import S3Gen
from ..models.s3gen.checkpoint_conversion import \
    convert_diffusers_transformer_keys

_ALLOWED_MISSING_SUBSTRINGS = (
    "_mel_filters",
    "window",
    "tokenizer._mel_filters",
    "tokenizer.window",
    "feature_extractor._mel_filters",
    "feature_extractor.window",
    "token_encoder.",
    "real_stft.",
    "real_istft.",
)


def _check_missing_keys(missing: list[str]) -> None:
    unexpected_missing = [
        key
        for key in missing
        if not any(fragment in key for fragment in _ALLOWED_MISSING_SUBSTRINGS)
    ]
    if unexpected_missing:
        raise RuntimeError(
            f"Checkpoint is missing required model keys: {unexpected_missing[:32]}"
        )


def prepare_export_safe_positional_encoding(
    model: torch.nn.Module, max_positions: int = 6144
) -> None:
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    probe = torch.zeros(1, max_positions, 512, device=device, dtype=dtype)
    for module in model.modules():
        if module.__class__.__name__ == "EspnetRelPositionalEncoding":
            module.extend_pe(probe)
        elif hasattr(module, "max_len") and hasattr(module, "pe"):
            if getattr(module, "max_len", 0) < max_positions and torch.is_tensor(
                getattr(module, "pe", None)
            ):
                # Absolute encoders in this model are not used for the VC token path,
                # but keep them on the target device when present.
                module.pe = module.pe.to(device=device)


def load_torch_model(
    checkpoint_dir: Path, device: str = "cpu", max_positions: int = 6144
) -> S3Gen:
    checkpoint_dir = Path(checkpoint_dir)
    model = S3Gen(meanflow=True)
    state = load_file(checkpoint_dir / "s3gen_meanflow.safetensors")
    state = convert_diffusers_transformer_keys(state)
    incompatible = model.load_state_dict(state, strict=False)
    _check_missing_keys(list(incompatible.missing_keys))
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint contains unexpected keys: {list(incompatible.unexpected_keys)[:32]}"
        )
    model.to(device).eval()
    model.mel2wav.optimize_for_inference()
    prepare_export_safe_positional_encoding(model, max_positions=max_positions)
    return model
