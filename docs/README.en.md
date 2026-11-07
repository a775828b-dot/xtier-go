# XTier Go

[中文](../README.md) · [Docs (Chinese)](README.md)

XTier Go is the Android branch of [XTier](https://github.com/a775828b-dot/xtier): an app for phones and tablets that
opens on a launcher, starts an inference server on the device with the chosen model and options, and serves an
OpenAI-compatible API on `http://127.0.0.1:18400/v1` for other apps (such as DSH Go X). The inference core is its own
C++ engine (SGLang / vLLM do not run on the phone); models are ported from vLLM main and checked against vLLM on a
server.

**Status: early development; short CPU requests now run on the Y900.** The installed app passes kernel/parser,
file verification, stop/restart and existing HTTP API checks. The checkpoint was verified through a temporary USB
transfer endpoint; direct mirror access timed out on the current network. DSH client integration, long contexts,
sustained background operation and GPU inference remain unverified. See the [device report](reports/2026-11-07-y900-bringup.md).

- **vLLM's semantics**: Dory's forward is ported operator by operator from vLLM main (PR #60149); config.json and
  the tokenizer's pre-tokenizer are checked at load, unknown layouts are refused.
- **Weights as they are**: NVFP4 (E2M1 codes, E4M3 block scales) computed straight from the mapped safetensors, no
  requantization; twice the E2M1 magnitudes are integers, one TBL lookup on ARM.
- **FP8 KV cache**: one scale per position and head, address space reserved and memory committed as the context
  grows; up to 256K.
- **Compatible API**: OpenAI chat (streaming, tool calls, reasoning content, cached-token usage) plus SGLang's
  `/generate`, `/flush_cache`, `/metrics`, so XTier's bench scripts run unchanged.

## Layout

| Directory | Contents |
|---|---|
| `engine/` | thread pool, safetensors mapping, kernels (portable + NEON), KV cache (linear / ring, FP8 / BF16), sampler, model pack interface |
| `packs/dory/` | Dory: input / recurrent / output segments, per_loop / last_loop KV, chunked prefill, sliding-window snapshots |
| `server/` | HTTP, OpenAI + SGLang subset, tokenizer, chat template (minja), streaming reasoning / tool-call parser, prefix cache, C API |
| `app/` | Android app (Kotlin, Jetpack Compose): launcher, foreground service in its own process, model downloads, JNI |
| `configs/` | one launcher card per model x device (card.json, the fields of XTier's console.py) |
| `reference/` | server-side references: plain PyTorch Dory, vLLM per-position logprobs and per-layer outputs, transformers-style template rendering |

## Quick start (Linux)

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -C build
build/kernels_test && build/parsers_test
build/xtg-server --model /path/to/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4 --threads 32 --opt context=131072
```

## Checks on the server (2026-11-07)

Tokenizer identical to Hugging Face tokenizers on 203 texts; chat template identical to transformers' jinja2
rendering on 10 cases; logits within 0.0004 of an FP32 PyTorch reference; NLL equal to vLLM 0.33.0 on English and
Chinese text (1.7443 vs 1.7451, 3.0066 vs 3.0068); decode path 48/48 tokens equal to a full recompute; API checks
(streamed tool calls, prefix reuse across turns, /generate, /flush_cache) pass; XTier's copy test gives exactly
vLLM's result. Details: [the report](reports/2026-11-07-host-bringup.md) (Chinese).

License: Apache-2.0. Third-party code: see NOTICE. Model weights: NVIDIA's OpenMDW license, not in this repository.
