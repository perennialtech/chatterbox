from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio as ta

from chatterbox.audio import DEC_COND_LEN, S3_SR, S3GEN_SR, resample_audio
from chatterbox.onnx_export.constants import (GRAPH_FLOW_DECODER_MEANFLOW2,
                                              GRAPH_REFERENCE_MEL_24K,
                                              GRAPH_S3_TOKENIZER_QUANTIZER,
                                              GRAPH_SPEAKER_ENCODER,
                                              GRAPH_TOKEN_TO_MU,
                                              GRAPH_VOCODER_HIFT)
from chatterbox.onnx_export.runtime import OnnxSessions
from chatterbox.vc.conditioning import VoiceConditionTensors
from chatterbox.vc.postprocess import apply_initial_trim_fade
from chatterbox.vc.preprocess import (compute_fbank, compute_s3_log_mel,
                                      load_wav_16k, load_wav_24k)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Chatterbox VC by explicitly orchestrating ONNX graphs."
    )
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime provider list.",
    )
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def _providers(value: str) -> list[str]:
    return [provider.strip() for provider in value.split(",") if provider.strip()]


def _as_int32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array.astype(np.int32, copy=False))


def _as_float32(array: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(array.astype(np.float32, copy=False))


def tokenize(runners, audio_16k: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    log_mel, mel_lengths = compute_s3_log_mel(audio_16k)
    out = runners[GRAPH_S3_TOKENIZER_QUANTIZER].run(
        {
            "log_mel": _as_float32(log_mel),
            "mel_lengths": _as_int32(mel_lengths),
        }
    )
    return out["speech_tokens"].astype(np.int64), out["speech_token_lengths"].astype(
        np.int64
    )


def load_reference_audio(path: Path) -> tuple[np.ndarray, np.ndarray]:
    ref_wav_24k = load_wav_24k(path, "cpu", max_len=DEC_COND_LEN)
    ref_wav_16k = resample_audio(ref_wav_24k, S3GEN_SR, S3_SR, "cpu")
    return (
        np.ascontiguousarray(ref_wav_24k.numpy().astype(np.float32)),
        np.ascontiguousarray(ref_wav_16k.numpy().astype(np.float32)),
    )


def extract_target_voice(runners, target_path: Path) -> VoiceConditionTensors:
    ref_wav_24k, ref_wav_16k = load_reference_audio(target_path)

    mel_out = runners[GRAPH_REFERENCE_MEL_24K].run({"wav_24k": ref_wav_24k})
    prompt_feat = mel_out["prompt_feat"].astype(np.float32)
    prompt_feat_len = mel_out["prompt_feat_len"].astype(np.int64)

    fbank, _ = compute_fbank(torch.from_numpy(ref_wav_16k))
    speaker_out = runners[GRAPH_SPEAKER_ENCODER].run({"fbank": _as_float32(fbank)})
    embedding = speaker_out["embedding"].astype(np.float32)

    prompt_token, prompt_token_len = tokenize(runners, torch.from_numpy(ref_wav_16k))

    if prompt_feat.shape[1] != 2 * prompt_token.shape[1]:
        target_len = prompt_feat.shape[1] // 2
        prompt_token = prompt_token[:, :target_len]
        prompt_token_len = np.array([target_len], dtype=np.int64)

    return VoiceConditionTensors.from_mapping(
        {
            "prompt_token": prompt_token,
            "prompt_token_len": prompt_token_len,
            "prompt_feat": prompt_feat,
            "prompt_feat_len": prompt_feat_len,
            "embedding": embedding,
        }
    )


def convert(
    sessions: OnnxSessions,
    runners,
    source_path: Path,
    target_voice: VoiceConditionTensors,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    source_16k = load_wav_16k(source_path, "cpu")
    speech_tokens, speech_token_lens = tokenize(runners, source_16k)

    token_out = runners[GRAPH_TOKEN_TO_MU].run(
        {
            "prompt_token": _as_int32(target_voice.prompt_token),
            "prompt_token_len": _as_int32(target_voice.prompt_token_len),
            "speech_token": _as_int32(speech_tokens),
            "speech_token_len": _as_int32(speech_token_lens),
            "embedding": _as_float32(target_voice.embedding),
        }
    )

    mu = _as_float32(token_out["mu"])
    mask = _as_float32(token_out["mask"])
    spks = _as_float32(token_out["spks"])
    prompt_mels = int(token_out["prompt_mel_len"].max())
    output_mels = int(token_out["output_mel_len"].max())

    cond = np.zeros_like(mu, dtype=np.float32)
    cond[:, :, :prompt_mels] = target_voice.prompt_feat[:, :prompt_mels, :].transpose(
        0, 2, 1
    )

    noise = rng.standard_normal(mu.shape).astype(np.float32)
    flow_out = runners[GRAPH_FLOW_DECODER_MEANFLOW2].run(
        {
            "noise": noise,
            "mask": mask,
            "mu": mu,
            "spks": spks,
            "cond": cond,
        }
    )
    mel = flow_out["mel"][:, :, prompt_mels : prompt_mels + output_mels].astype(
        np.float32
    )

    constants = sessions.manifest["constants"]
    source_hop = int(constants["source_hop"])
    source_phase = np.zeros((mel.shape[0], 9, 1), dtype=np.float32)
    source_noise = rng.standard_normal(
        (mel.shape[0], 9, mel.shape[2] * source_hop)
    ).astype(np.float32)

    vocoder_out = runners[GRAPH_VOCODER_HIFT].run(
        {
            "speech_feat": mel,
            "source_phase": source_phase,
            "source_noise": source_noise,
        }
    )
    wav = vocoder_out["wav"].astype(np.float32)
    apply_initial_trim_fade(wav, int(constants["trim_fade_len"]))
    return wav


def main() -> None:
    args = parse_args()
    sessions = OnnxSessions.from_artifact_dir(
        args.artifact_dir,
        precision=args.precision,
        providers=_providers(args.providers),
    )
    runners = {
        name: sessions.runner(name)
        for name in (
            GRAPH_REFERENCE_MEL_24K,
            GRAPH_SPEAKER_ENCODER,
            GRAPH_S3_TOKENIZER_QUANTIZER,
            GRAPH_TOKEN_TO_MU,
            GRAPH_FLOW_DECODER_MEANFLOW2,
            GRAPH_VOCODER_HIFT,
        )
    }

    target_voice = extract_target_voice(runners, args.target)
    wav = convert(sessions, runners, args.source, target_voice, seed=args.seed)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ta.save(str(args.output), torch.from_numpy(wav), S3GEN_SR)
    print(f"wrote={args.output} sample_rate={S3GEN_SR}")


if __name__ == "__main__":
    main()
