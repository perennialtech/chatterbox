from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

MODEL_OUT = "model.safetensors"
CONFIG_OUT = "config.json"
REFERENCE_OUT = "builtin_reference.safetensors"


def convert_model_state(old: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    new = {}

    for key, value in old.items():
        if key in {"trim_fade"}:
            continue
        if key.startswith("tokenizer._mel_filters") or key.startswith(
            "tokenizer.window"
        ):
            continue

        new_key = key
        new_key = replace_prefix(new_key, "tokenizer.", "tokenizer.backend.")
        new_key = replace_prefix(
            new_key, "speaker_encoder.", "reference_encoder.speaker_encoder."
        )
        new_key = replace_prefix(
            new_key, "flow.input_embedding.", "mel_generator.input_embedding."
        )
        new_key = replace_prefix(
            new_key, "flow.spk_embed_affine_layer.", "mel_generator.speaker_projection."
        )
        new_key = replace_prefix(new_key, "flow.encoder.", "mel_generator.encoder.")
        new_key = replace_prefix(
            new_key, "flow.encoder_proj.", "mel_generator.encoder_projection."
        )
        new_key = replace_prefix(
            new_key, "flow.decoder.estimator.", "mel_generator.sampler.estimator."
        )
        new_key = replace_prefix(new_key, "mel2wav.", "vocoder.")

        new[new_key] = value

    return new


def replace_prefix(key: str, old: str, new: str) -> str:
    if key.startswith(old):
        return new + key[len(old) :]
    return key


def convert_reference(old_checkpoint_dir: Path) -> dict[str, torch.Tensor] | None:
    conds_path = old_checkpoint_dir / "conds.pt"
    if not conds_path.exists():
        return None

    states = torch.load(conds_path, map_location="cpu")
    ref = states["gen"]

    prompt_tokens = ref["prompt_token"].long()
    prompt_token_lengths = ref["prompt_token_len"].long()
    prompt_mels = ref["prompt_feat"]
    if prompt_mels.ndim != 3:
        raise ValueError("Expected old prompt_feat shape [B, T, 80].")

    prompt_mels = prompt_mels.transpose(1, 2).contiguous()
    prompt_mel_lengths = prompt_token_lengths * 2

    return {
        "prompt_tokens": prompt_tokens,
        "prompt_token_lengths": prompt_token_lengths,
        "prompt_mels": prompt_mels,
        "prompt_mel_lengths": prompt_mel_lengths,
        "speaker_embedding": ref["embedding"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-checkpoint-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    old_dir = args.old_checkpoint_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    old_weights = old_dir / "s3gen_meanflow.safetensors"
    if not old_weights.exists():
        raise FileNotFoundError(old_weights)

    old_state = load_file(old_weights, device="cpu")
    new_state = convert_model_state(old_state)
    save_file(new_state, out_dir / MODEL_OUT)

    config = {
        "tokenizer_sr": 16_000,
        "sample_rate": 24_000,
        "max_reference_seconds": 10,
        "model_revision": 2,
    }
    (out_dir / CONFIG_OUT).write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8"
    )

    ref = convert_reference(old_dir)
    if ref is not None:
        save_file(ref, out_dir / REFERENCE_OUT)


if __name__ == "__main__":
    main()
