# Voice conversion latency

This evaluation targets short, warmed voice conversion calls, not an end-to-end HTTP or TTS service. Chatterbox returns a complete waveform, so conversion wall time is also its time to first audio. The model keeps the same checkpoint, tokenizer, non-causal context, and two flow steps; no training or streaming approximation is involved.

## Method

The baseline is commit `b9940717c1e8a31bda54baffb2f21ccf170f23d1`. Both revisions run with compilation enabled on the same RTX 3080, using PyTorch `2.14.0+cu130`, Python 3.11.15, and two PyTorch CPU threads. Each disposable container has four CPU cores available. The running production services remain untouched; benchmark containers run sequentially.

The source is cropped from `espeak.wav` to 1, 3, 5, and 10 seconds. The reference is byte-identical to `voices/default_voice.wav` in the TTS project (SHA-256 `38f177eefe6df0c2ad04086694569448f7f3cba7c082fd28ba25a78586329ff2`). Each length has a separate initial call, three warmup calls, and 30 timed calls. Timings include completion of the CPU waveform copy, not just GPU kernel submission.

Artifact fingerprints:

- Checkpoint SHA-256: `d65cb687a2ed581ee6cc297e919ffefa63386944f42364ae13b78a594945514f`.
- Source SHA-256: `b0d844369b6090eee9afea4b36367ac390277328eb0334943bdfe7fe5e54eb44`.
- Container image ID: `cbe884a55883200205237fd5dbd1d3587c9b1a825eb128e66b80aa903760766a`; NVIDIA driver: `595.99.02`.

File mode measures source decoding/resampling plus repeated `target_voice_path` calls. Tensor mode prepares the requested reference outside the timed calls and supplies 16 kHz audio tensors. This distinction separates the reference-cache benefit from faster flow execution.

See [the contributor guide](../CONTRIBUTING.md) for the benchmark and precision-comparison commands.

## Warmed results

File-based conversion measurements on 2026-10-21:

| Source length | Baseline median / p95 | Optimized median / p95 | Median reduction |
| --- | --- | --- | --- |
| 1 s | 120.1 / 126.7 ms | 51.3 / 55.0 ms | 57% |
| 3 s | 126.6 / 130.3 ms | 58.4 / 60.0 ms | 54% |
| 5 s | 132.5 / 137.5 ms | 63.5 / 66.7 ms | 52% |
| 10 s | 161.3 / 165.0 ms | 80.7 / 84.5 ms | 50% |

Prepared-reference tensor calls improve by 27–30%: at three seconds, median latency falls from 73.6 to 53.1 ms; at ten seconds, from 107.5 to 75.1 ms. The larger file-mode gain includes eliminating repeated reference preparation.

The three-second file-mode ablation shows why the optimizations are combined:

| Configuration | Median latency |
| --- | --- |
| Baseline | 126.6 ms |
| Reference cache, FP32, no graphs | 81.5 ms |
| Reference cache, FP16, no graphs | 77.7 ms |
| Reference cache, FP16, flow graphs | 58.4 ms |

Compiler caches from earlier experiments were retained. Even so, initial file-mode calls took 54.8 seconds on the baseline and 58.6 seconds on the optimized revision. The explicit target warmup took 64.4 seconds; subsequent initial calls for each timed length took 54–81 ms. These are observed initialization costs with existing compiler caches, not clean-cache cold-start benchmarks.

## Memory accounting

PyTorch peak allocated/reserved counters cover the entire benchmark process, including loading and warmup. Separate `nvidia-smi` samples at roughly 100 ms intervals measure the benchmark process's VRAM, including its CUDA context. These process measurements exclude the already-running production service, and do not measure the combined Kokoro-plus-Chatterbox deployment.

An additional run preloads the actual target and calls the public `warmup()` method. This includes longer flow/vocoder shapes than the timed clips, so its peak is relevant to service startup as well as short-input inference. The benchmark allocator cap is 5.5 GiB; it is a safety limit, not a claim about actual usage.

| Run | Sampled peak process VRAM | Peak PyTorch reserved |
| --- | --- | --- |
| Baseline, 1–10 s | 1,588 MiB (1.55 GiB) | 1.28 GiB |
| Optimized, 1–10 s | 1,842 MiB (1.80 GiB) | 1.45 GiB |
| Optimized, including public warmup | 2,264 MiB (2.21 GiB) | 1.80 GiB |
| Optimized, 12 serialized workers | 2,966 MiB (2.90 GiB) | 2.12 GiB |

The worker check matches the TTS service's serialized `to_thread` conversion pattern. Twelve distinct `ThreadPoolExecutor` workers make three calls each under one shared lock, rotating through 0.25, 1, 3, 5, 10, and 10.12 second inputs. It starts without main-thread flow warmup, so different workers can initialize new shapes. All calls produce finite audio with the expected token-frame duration. Worker changes increase memory use, but the measured peak remains below 4 GiB. This is not a concurrent-conversion benchmark: calls on one converter must remain serialized.

## Quality and scope

The precision check uses identical source tokens and seeded FP32 flow noise, with the same reference, vocoder, and flow-step count. It compares finite outputs, exact output lengths, mel error, waveform signal-to-noise ratio, and speaker-vector cosine similarity across several speech sources, including a clip just above the ten-second flow-chunk boundary. FP16 execution is approximate and is not bit-identical to FP32. These numeric checks do not establish perceptual equivalence or replace listening tests.

The eleven checked cases have maximum relative mel L2 error of 0.034%, maximum mel RMSE of 0.00168, minimum speaker cosine similarity of 0.99991, and waveform SNR of 29–44 dB. All outputs are finite and retain the FP32 output length. Sources include eSpeak speech and two neural speech recordings; this is a limited regression check, not a broad quality benchmark.

First calls and unseen shapes can spend substantial time compiling. Warmed timing results must not be read as cold-start guarantees. Preloading a reference and warming its representative shapes moves that work before request serving, but cannot cover every possible shape or target voice.

## Options not selected

True incremental conversion changes the future context seen by the tokenizer and flow model. This evaluation keeps the offline behavior rather than introducing chunk seams or relying on retraining.

Compiler freezing was tested, but its small full-call improvement in exploratory runs did not justify additional stored weight copies and compilation cost. A forced cuDNN attention backend was slower than the existing memory-efficient backend for the decoder's masked attention. FlashAttention rejected the non-null attention mask. None of these alternatives is enabled.

## Avoiding redundant evaluation-mode traversals (2026-10-24)

Calling `eval()` on the 2,517-module S3Gen instance takes approximately 4.5 ms, even when it is already in evaluation mode. The inference methods now make the transition only when the top-level model is in training mode. A subsequent `train()` call still causes the next inference to switch back to evaluation.

Warmed, file-based conversion on an RTX 3080 with PyTorch 2.14.1+cu130, compiled flow/vocoder, two CPU threads, and three warmup plus 30 measured calls per length. Both runs use the same eSpeak recording as **both** source and reference, rather than the production reference used in the earlier benchmark. The shared compiler cache was warmed before timing, and production services were not changed.

| Source length | PR #1 master median | With conditional `eval()` | Median reduction |
| --- | --- | --- | --- |
| 1 s | 48.0 ms | 42.9 ms | 10.6% |
| 3 s | 55.7 ms | 50.4 ms | 9.6% |
| 5 s | 60.2 ms | 55.3 ms | 8.2% |
| 10 s | 79.3 ms | 74.6 ms | 6.0% |

Peak reserved PyTorch memory was 1.95 GiB in both runs. These figures exclude service overhead and cold-start compilation. An ablation of additional tokenization, conditioning, and vocoder micro-optimizations did not improve the full conversion results and was not retained.
