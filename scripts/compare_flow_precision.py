"""Compare seeded FP32 and FP16 flow outputs on a CUDA host."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from chatterbox.audio import S3_SR, load_audio_mono, resample_audio
from chatterbox.models.speaker.features import extract_fbank_features
from chatterbox.vc.backends.torch_backend import TorchVCBackend


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--source", type=Path, action="append", required=True)
    parser.add_argument(
        "--seconds", type=float, nargs="+", default=[1, 3, 5, 10, 10.12]
    )
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("precision comparison requires CUDA")
    if any(not math.isfinite(s) or s <= 0 for s in args.seconds):
        parser.error("seconds must be finite and positive")

    torch.set_num_threads(2)
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, 5.5 * 2**30 / total))
    os.environ["CHATTERBOX_FLOW_FP16"] = "0"
    backend = TorchVCBackend.from_local(args.checkpoint, "cuda", compile=args.compile)
    backend.set_target_voice_path(args.reference)
    model = backend.s3gen
    cases = []
    for path in args.source:
        audio = load_audio_mono(path, S3_SR, "cuda")
        # Short files contribute their full length as well as shorter crops.
        lengths = {min(round(s * S3_SR), audio.numel()) for s in args.seconds}
        for length in sorted(lengths):
            tokens, lens = model.tokenizer(audio[:length].unsqueeze(0))
            cases.append((str(path), length / S3_SR, tokens, lens))

    def generate(tokens, lens):
        mel = model.flow_inference(
            tokens, speech_token_lens=lens, ref_condition=backend.ref_condition
        )
        wav, _ = model.hift_inference(mel.to(dtype=model.dtype))
        wav = wav[:, : mel.size(-1) * model.mel2wav.source_hop].contiguous()
        model._apply_output_fades(wav)
        return wav, mel

    def speaker(wav):
        audio = resample_audio(wav, backend.sr, S3_SR, "cuda")
        return model.speaker_encoder(extract_fbank_features(audio)).cpu()

    baseline = []
    rows = []
    for precision in ["0", "1"]:
        os.environ["CHATTERBOX_FLOW_FP16"] = precision
        for index, (path, seconds, tokens, lens) in enumerate(cases):
            for _ in range(3):
                generate(tokens, lens)
            torch.manual_seed(7654)
            wav, mel = generate(tokens, lens)
            embedding = speaker(wav)
            wav, mel = wav.cpu(), mel.float().cpu()
            if precision == "0":
                baseline.append((wav, mel, embedding))
                continue
            original, original_mel, original_embedding = baseline[index]
            if wav.shape != original.shape or not torch.isfinite(wav).all():
                raise RuntimeError(f"invalid FP16 output for {path} at {seconds}s")
            mse = (wav - original).square().mean().item()
            row = {
                "source": path,
                "seconds": seconds,
                "output_samples": wav.numel(),
                "mel_rmse": (mel - original_mel).square().mean().sqrt().item(),
                "mel_relative_l2": (
                    torch.linalg.vector_norm(mel - original_mel)
                    / torch.linalg.vector_norm(original_mel).clamp_min(1e-12)
                ).item(),
                "waveform_snr_db": (
                    10 * math.log10(max(original.square().mean().item(), 1e-30) / mse)
                    if mse > 0
                    else None
                ),
                "speaker_cosine": F.cosine_similarity(
                    embedding, original_embedding
                ).item(),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
            if row["mel_relative_l2"] > 0.01 or row["speaker_cosine"] < 0.99:
                raise RuntimeError(f"precision regression: {row}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
