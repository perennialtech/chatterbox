from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...audio import S3GEN_SR, SPEECH_VOCAB_SIZE
from ...models.s3gen.conditioning import S3ReferenceCondition
from ...models.s3gen.pipeline import FLOW_CHUNK_TOKENS, FLOW_CONTEXT_TOKENS
from ..artifacts import load_manifest
from ..constants import (GRAPH_REFERENCE_MEL_24K, GRAPH_S3_TOKENIZER_LOG_MEL,
                         GRAPH_S3_TOKENIZER_QUANTIZER, GRAPH_SPEAKER_ENCODER,
                         GRAPH_TOKEN_TO_MU, GRAPH_VOCODER_HIFT)
from ..errors import OnnxValidationError
from ..graph_spec import GraphSpec
from ..graphs import ALL_GRAPHS
from ..model_loading import load_torch_model
from ..runtime import OnnxS3Gen
from .metrics import compare_cosine, compare_exact, compare_tensors, to_numpy
from .tolerances import CosineTolerance, Tolerance, tolerance_for_graph


def _ort_providers(ort, device: str) -> list[str]:
    available = set(ort.get_available_providers())
    if str(device).startswith("cuda") and "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _run_ort(
    path: Path,
    input_names: list[str],
    inputs: tuple[torch.Tensor, ...],
    device: str = "cpu",
) -> list[np.ndarray]:
    import onnxruntime as ort

    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(
        str(path),
        sess_options=session_options,
        providers=_ort_providers(ort, device),
    )
    actual_input_names = {inp.name for inp in session.get_inputs()}

    feed = {
        name: np.ascontiguousarray(t.detach().cpu().numpy())
        for name, t in zip(input_names, inputs)
        if name in actual_input_names
    }

    return session.run(None, feed)


def _bucket_token_length(length: int, buckets: tuple[int, ...]) -> int:
    for bucket in buckets:
        if length <= bucket:
            return bucket
    raise ValueError(f"length {length} exceeds largest token bucket {buckets[-1]}")


def _validation_cases_for_spec(
    spec: GraphSpec, manifest: dict
) -> list[tuple[torch.Tensor, ...]]:
    name = spec.name
    if name == GRAPH_S3_TOKENIZER_LOG_MEL:
        return [spec.make_dummy_inputs(samples=samples) for samples in (16000, 32123)]
    if name == GRAPH_S3_TOKENIZER_QUANTIZER:
        return [spec.make_dummy_inputs(mel_frames=frames) for frames in (64, 257, 3000)]
    if name == GRAPH_SPEAKER_ENCODER:
        return [spec.make_dummy_inputs(frames=frames) for frames in (80, 256, 501)]
    if name == GRAPH_REFERENCE_MEL_24K:
        return [spec.make_dummy_inputs(samples=samples) for samples in (24000, 48000)]

    if name.startswith(f"{GRAPH_TOKEN_TO_MU}_"):
        token_bucket = int(name.rsplit("_", 1)[1].removesuffix("tok"))
        token_buckets = tuple(
            int(x) for x in manifest["constants"]["token_to_mu_token_buckets"]
        )
        cases = []
        prompt_tokens = 25
        for speech_tokens in (32, 250, 251, 384):
            if (
                _bucket_token_length(prompt_tokens + speech_tokens, token_buckets)
                == token_bucket
            ):
                token = torch.zeros(1, token_bucket, dtype=torch.int32)
                token[:, :prompt_tokens] = 3
                token[:, prompt_tokens : prompt_tokens + speech_tokens] = 4
                cases.append(
                    (
                        token,
                        torch.tensor([prompt_tokens], dtype=torch.int32),
                        torch.tensor([speech_tokens], dtype=torch.int32),
                        torch.randn(1, 192, dtype=torch.float32),
                    )
                )
        return cases or [spec.make_dummy_inputs()]

    if name.startswith("flow_decoder_meanflow2_"):
        return [spec.make_dummy_inputs()]

    if name.startswith(f"{GRAPH_VOCODER_HIFT}_"):
        mel_bucket = int(name.rsplit("_", 1)[1].removesuffix("mel"))
        requested = {1, 64, 251}
        if mel_bucket in requested:
            return [spec.make_dummy_inputs()]
        return [spec.make_dummy_inputs()]

    return [spec.make_dummy_inputs()]


def _compare_padded_token_ids(
    name: str,
    expected_tokens,
    actual_tokens,
    expected_lengths,
) -> dict:
    expected_np = to_numpy(expected_tokens)
    actual_np = np.asarray(actual_tokens)
    lengths_np = to_numpy(expected_lengths).astype(np.int64).reshape(-1)

    if expected_np.ndim != 2 or actual_np.ndim != 2:
        raise OnnxValidationError(
            f"{name} must compare rank-2 token tensors, got {expected_np.shape} and {actual_np.shape}"
        )
    if expected_np.shape[0] != actual_np.shape[0]:
        raise OnnxValidationError(
            f"{name} batch mismatch: expected {expected_np.shape[0]}, actual {actual_np.shape[0]}"
        )
    if lengths_np.shape != (expected_np.shape[0],):
        raise OnnxValidationError(
            f"{name} length tensor shape mismatch: expected {(expected_np.shape[0],)}, actual {lengths_np.shape}"
        )
    if np.any(lengths_np < 0):
        raise OnnxValidationError(f"{name} contains negative sequence lengths")

    max_len = int(lengths_np.max()) if lengths_np.size else 0
    if max_len > expected_np.shape[1] or max_len > actual_np.shape[1]:
        raise OnnxValidationError(
            f"{name} length {max_len} exceeds token widths: expected {expected_np.shape[1]}, actual {actual_np.shape[1]}"
        )

    valid_tokens = 0
    for batch_index, length in enumerate(lengths_np):
        length = int(length)
        valid_tokens += length
        expected_prefix = expected_np[batch_index, :length]
        actual_prefix = actual_np[batch_index, :length]
        if not np.array_equal(expected_prefix, actual_prefix):
            mismatch = np.flatnonzero(expected_prefix != actual_prefix)
            token_index = int(mismatch[0])
            expected_value = expected_prefix[token_index]
            actual_value = actual_prefix[token_index]
            if hasattr(expected_value, "item"):
                expected_value = expected_value.item()
            if hasattr(actual_value, "item"):
                actual_value = actual_value.item()
            raise OnnxValidationError(
                f"{name} valid-token parity failed: {int(mismatch.size)} mismatched values "
                f"in batch {batch_index}; first mismatch at token {token_index}: "
                f"expected {expected_value}, actual {actual_value}"
            )

    return {
        "exact_valid_prefix": True,
        "valid_tokens": int(valid_tokens),
        "expected_padding_tokens": int(expected_np.size - valid_tokens),
        "actual_padding_tokens": int(actual_np.size - valid_tokens),
    }


def _compare_quantizer_outputs(
    spec_name: str,
    output_names: list[str],
    torch_outputs: tuple,
    ort_outputs: list[np.ndarray],
) -> dict:
    expected = dict(zip(output_names, torch_outputs))
    actual = dict(zip(output_names, ort_outputs))

    if "speech_tokens" not in expected or "speech_token_lengths" not in expected:
        raise OnnxValidationError(
            f"{spec_name} validation requires speech_tokens and speech_token_lengths outputs"
        )

    report = {
        "speech_token_lengths": compare_exact(
            f"{spec_name}.speech_token_lengths",
            expected["speech_token_lengths"],
            actual["speech_token_lengths"],
        )
    }
    report["speech_tokens"] = _compare_padded_token_ids(
        f"{spec_name}.speech_tokens",
        expected["speech_tokens"],
        actual["speech_tokens"],
        expected["speech_token_lengths"],
    )
    return report


def _compare_output(spec_name: str, output_name: str, expected, actual) -> dict:
    if not torch.is_floating_point(expected):
        return compare_exact(f"{spec_name}.{output_name}", expected, actual)
    tolerance = tolerance_for_graph(spec_name)
    if isinstance(tolerance, CosineTolerance):
        return compare_cosine(f"{spec_name}.{output_name}", expected, actual, tolerance)
    assert isinstance(tolerance, Tolerance)
    return compare_tensors(f"{spec_name}.{output_name}", expected, actual, tolerance)


def _run_graph_validation(
    artifact_dir: Path,
    manifest: dict,
    model,
    device: str,
) -> dict:
    report = {}

    for spec in ALL_GRAPHS:
        graph_entry = manifest["graphs"][spec.name]
        onnx_path = artifact_dir / graph_entry["path"]
        module = spec.make_module(model).to(device).eval()
        graph_cases = []

        for case_index, inputs in enumerate(_validation_cases_for_spec(spec, manifest)):
            inputs = tuple(x.to(device) for x in inputs)
            with torch.inference_mode():
                torch_outputs = module(*inputs)
            if not isinstance(torch_outputs, (tuple, list)):
                torch_outputs = (torch_outputs,)

            ort_outputs = _run_ort(onnx_path, spec.input_names, inputs, device=device)
            if spec.name == GRAPH_S3_TOKENIZER_QUANTIZER:
                case_report = _compare_quantizer_outputs(
                    spec.name,
                    spec.output_names,
                    torch_outputs,
                    ort_outputs,
                )
            else:
                case_report = {}
                for output_name, expected, actual in zip(
                    spec.output_names, torch_outputs, ort_outputs
                ):
                    case_report[output_name] = _compare_output(
                        spec.name, output_name, expected, actual
                    )
            graph_cases.append({"case_index": case_index, "outputs": case_report})

        report[spec.name] = {"cases": graph_cases}

    return report


def _make_ref_dict(condition: S3ReferenceCondition) -> dict[str, np.ndarray]:
    return {
        "prompt_token": condition.prompt_token.detach().cpu().numpy().astype(np.int32),
        "prompt_token_len": condition.prompt_token_len.detach()
        .cpu()
        .numpy()
        .astype(np.int32),
        "prompt_feat": condition.prompt_feat.detach().cpu().numpy().astype(np.float32),
        "embedding": condition.embedding.detach().cpu().numpy().astype(np.float32),
    }


def _torch_flow_window(
    model,
    *,
    window: torch.Tensor,
    window_len: torch.Tensor,
    ref_condition: S3ReferenceCondition,
    noise_np: np.ndarray,
) -> torch.Tensor:
    prompt_len = ref_condition.prompt_token.size(1)
    padded_window = model._pad_tokens_to_bucket(window, prompt_len)

    flow = model.flow
    embedding = F.normalize(ref_condition.embedding, dim=1)
    spks = flow.spk_embed_affine_layer(embedding)

    token = torch.cat([ref_condition.prompt_token, padded_window], dim=1)
    token_len = ref_condition.prompt_token_len + window_len
    token_mask = ~model.flow.encoder.__class__.__mro__[0].__module__
    del token_mask

    from ...models.s3gen.utils.mask import make_pad_mask

    mask = (~make_pad_mask(token_len, max_len=token.size(1))).unsqueeze(-1).to(spks)
    embedded = flow.input_embedding(token.long()) * mask
    h, h_masks = flow.encoder(embedded, token_len)
    h_lengths = h_masks.sum(dim=-1).squeeze(dim=-1)
    mu = flow.encoder_proj(h).transpose(1, 2).contiguous()

    prompt_mel_len = ref_condition.prompt_feat.size(1)
    cond = torch.zeros(
        [1, mu.size(2), flow.output_size],
        device=mu.device,
        dtype=mu.dtype,
    )
    cond[:, :prompt_mel_len] = ref_condition.prompt_feat
    cond = cond.transpose(1, 2).contiguous()

    flow_mask = (~make_pad_mask(h_lengths, max_len=h.shape[1])).unsqueeze(1).to(mu)

    noise = torch.from_numpy(noise_np).to(device=mu.device, dtype=mu.dtype)
    if noise.shape[2] != mu.shape[2]:
        noise = noise[:, :, : mu.shape[2]].contiguous()

    t_span = torch.tensor([0.0, 0.5, 1.0], device=mu.device, dtype=mu.dtype)
    mel = flow.decoder.decode_from_noise(
        x=noise,
        mu=mu,
        mask=flow_mask,
        spks=spks,
        cond=cond,
        t_span=t_span,
    )
    return mel[
        :,
        :,
        prompt_mel_len : prompt_mel_len + window_len.item() * model._token_mel_ratio,
    ]


def _torch_deterministic_pipeline(
    model,
    speech_tokens: torch.Tensor,
    ref_condition: S3ReferenceCondition,
    flow_noise_callback,
    source_phase: np.ndarray,
    source_noise: np.ndarray,
    manifest: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    speech_tokens, speech_token_lens, original_mel_len = model._prepare_target_tokens(
        speech_tokens=speech_tokens,
        speech_token_lens=None,
    )

    target_token_len = int(speech_token_lens.max().detach().cpu())
    chunk_tokens = max(1, FLOW_CHUNK_TOKENS)
    context_tokens = max(FLOW_CONTEXT_TOKENS, model._final_context_token_count)
    token_mel_ratio = model._token_mel_ratio

    chunks = []
    for chunk_index, center_start in enumerate(
        range(0, target_token_len, chunk_tokens)
    ):
        center_end = min(center_start + chunk_tokens, target_token_len)
        left_start = max(0, center_start - context_tokens)
        right_end = min(target_token_len, center_end + context_tokens)

        window = speech_tokens[:, left_start:right_end].contiguous()
        window_len = right_end - left_start

        if center_end == target_token_len:
            window, window_len_tensor = model._append_silence_context(
                window,
                torch.tensor([window_len], dtype=torch.long, device=model.device),
                model._final_context_token_count,
            )
        else:
            window_len_tensor = torch.tensor(
                [window_len],
                dtype=torch.long,
                device=model.device,
            )

        total_tokens = ref_condition.prompt_token.size(1) + window.size(1)
        token_buckets = tuple(
            int(x) for x in manifest["constants"]["token_to_mu_token_buckets"]
        )
        token_bucket = _bucket_token_length(total_tokens, token_buckets)
        noise_shape = (1, 80, token_bucket * token_mel_ratio)
        noise_np = flow_noise_callback(chunk_index, noise_shape)

        window_mels = _torch_flow_window(
            model,
            window=window,
            window_len=window_len_tensor,
            ref_condition=ref_condition,
            noise_np=noise_np,
        )

        mel_start = (center_start - left_start) * token_mel_ratio
        mel_end = mel_start + (center_end - center_start) * token_mel_ratio
        chunks.append(window_mels[:, :, mel_start:mel_end].contiguous())

    output_mels = torch.cat(chunks, dim=-1)[:, :, :original_mel_len].contiguous()

    vocoder_mel_buckets = tuple(
        int(x) for x in manifest["constants"]["vocoder_mel_buckets"]
    )
    vocoder_bucket = next(
        bucket for bucket in vocoder_mel_buckets if output_mels.size(2) <= bucket
    )
    padded_mels = F.pad(output_mels, (0, vocoder_bucket - output_mels.size(2)))
    wav, source = model.mel2wav.inference(
        padded_mels.to(dtype=model.dtype),
        source_phase=torch.from_numpy(source_phase).to(
            device=model.device, dtype=model.dtype
        ),
        source_noise=torch.from_numpy(source_noise).to(
            device=model.device, dtype=model.dtype
        ),
    )

    original_samples = min(wav.size(1), original_mel_len * model.mel2wav.source_hop)
    wav = wav[:, :original_samples].contiguous()
    source = source[:, :, :original_samples].contiguous()
    model._apply_output_fades(wav)
    return output_mels, wav


def _run_full_pipeline_validation(
    artifact_dir: Path,
    manifest: dict,
    model,
    device: str,
) -> dict:
    onnx_model = OnnxS3Gen.from_artifact_dir(
        artifact_dir,
        providers=_ort_providers(__import__("onnxruntime"), device),
    )

    t = torch.linspace(0, 1, S3GEN_SR * 2, device=model.device)
    ref_wav = (0.1 * torch.sin(2 * math.pi * 220.0 * t)).unsqueeze(0)
    ref_condition = model.embed_ref(ref_wav, S3GEN_SR)
    ref_dict = _make_ref_dict(ref_condition)

    speech_tokens = (
        torch.arange(0, 530, device=model.device) % (SPEECH_VOCAB_SIZE - 1)
    ).long()

    def flow_noise_callback(chunk_index: int, shape: tuple[int, ...]) -> np.ndarray:
        rng = np.random.default_rng(1000 + chunk_index)
        return rng.standard_normal(shape).astype(np.float32)

    original_mel_len = int(speech_tokens.numel()) * int(
        manifest["constants"]["token_mel_ratio"]
    )
    vocoder_bucket = next(
        int(bucket)
        for bucket in manifest["constants"]["vocoder_mel_buckets"]
        if original_mel_len <= int(bucket)
    )
    source_hop = int(manifest["constants"]["source_hop"])
    harmonics = int(manifest["constants"]["vocoder_harmonics"])

    source_rng = np.random.default_rng(2000)
    source_phase = source_rng.uniform(
        -math.pi,
        math.pi,
        size=(1, harmonics, 1),
    ).astype(np.float32)
    source_phase[:, :1, :] = 0.0
    source_noise = source_rng.standard_normal(
        (1, harmonics, vocoder_bucket * source_hop)
    ).astype(np.float32)

    with torch.inference_mode():
        expected_mel, expected_wav = _torch_deterministic_pipeline(
            model,
            speech_tokens,
            ref_condition,
            flow_noise_callback,
            source_phase,
            source_noise,
            manifest,
        )

    actual_wav, _actual_source, actual_mel, debug = onnx_model.inference(
        speech_tokens.detach().cpu().numpy(),
        ref_dict,
        flow_noise_callback=flow_noise_callback,
        source_phase=source_phase,
        source_noise=source_noise,
        return_mel=True,
        return_debug=True,
    )

    for chunk in debug["chunks"]:
        expected_mel_start = (chunk["center_start"] - chunk["left_start"]) * int(
            manifest["constants"]["token_mel_ratio"]
        )
        if chunk["mel_start"] != expected_mel_start:
            raise AssertionError(
                "runtime chunk mel_start does not match context formula"
            )
    if debug["chunks"][-1]["appended_silence_tokens"] != int(
        manifest["constants"]["final_context_token_count"]
    ):
        raise AssertionError(
            "final chunk did not append the configured silence context"
        )
    if debug["concat_mel_len"] != debug["original_mel_len"]:
        raise AssertionError(
            "concatenated chunk mel length does not match original length"
        )

    mel_report = compare_tensors(
        "full_pipeline.mel",
        expected_mel,
        actual_mel,
        tolerance_for_graph("full_pipeline_mel"),
    )
    wav_report = compare_tensors(
        "full_pipeline.wav",
        expected_wav,
        actual_wav,
        tolerance_for_graph("full_pipeline_wav"),
    )
    return {
        "mel": mel_report,
        "wav": wav_report,
        "runtime_slicing": {
            "sliced_after_prompt": True,
            "context_crop_formula": True,
            "final_silence_context": True,
            "concat_mel_len": debug["concat_mel_len"],
            "original_mel_len": debug["original_mel_len"],
        },
    }


def run_validation_with_model(
    artifact_dir: Path,
    model,
    device: str = "cpu",
) -> dict:
    torch.manual_seed(1234)
    np.random.seed(1234)

    artifact_dir = Path(artifact_dir).resolve()
    manifest = load_manifest(artifact_dir)
    report: dict[str, dict] = {"graphs": {}}

    report["graphs"] = _run_graph_validation(
        artifact_dir=artifact_dir,
        manifest=manifest,
        model=model,
        device=device,
    )
    report["full_pipeline"] = _run_full_pipeline_validation(
        artifact_dir=artifact_dir,
        manifest=manifest,
        model=model,
        device=device,
    )

    out_dir = artifact_dir / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True)
    )
    (out_dir / "full_pipeline.json").write_text(
        json.dumps(report["full_pipeline"], indent=2, sort_keys=True)
    )
    return report


def run_validation(
    artifact_dir: Path,
    checkpoint_dir: Path,
    device: str = "cpu",
) -> dict:
    model = load_torch_model(Path(checkpoint_dir).resolve(), device=device)
    return run_validation_with_model(
        artifact_dir=Path(artifact_dir).resolve(),
        model=model,
        device=device,
    )
