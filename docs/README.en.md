# XTier Go

[中文](../README.md) · [Docs (Chinese)](README.md) · [NPU path (Chinese)](npu.md)

A 4B reasoning model running locally on an Android tablet, the whole step on the Qualcomm NPU. XTier Go is an app: it
opens on a launcher, you pick the model and a context bucket, and it serves an OpenAI-compatible API on
`http://127.0.0.1:18400/v1` for clients on the same device such as DSH Go X. The model is NVIDIA's
[Nemotron-Labs Dory 1.0 4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-BF16), a looped nGPT.

> **Status (2026-11-08): development paused.** Short conversations (the 4K and 8K buckets) work: decode 12–16 tok/s, 20–35 with prompt lookup. Long contexts are too slow (3.6 tok/s at 32K), so DSH Go X only works in its minimal mode. Every change to a Qualcomm static graph means a new export (40–60 minutes); iterating is too slow, and the NPU path stops here.

## Measured (Lenovo Y900, Snapdragon 8 Elite)

| Backend · bucket | prefill | decode | decode + prompt lookup |
|---|---:|---:|---|
| **NPU · 1 loop · 4K** | 342 tok/s | 15.5 tok/s | 20–35 (by task) |
| NPU · 1 loop · 8K | 162 | 12.5 | about 15 (DSH, measured) |
| NPU · 1 loop · 32K | 54 | 3.6 | — |
| NPU · 2 loops · 4K | 270–310 | 9.4 | 13 |
| GPU (Adreno 830, fallback) | 42 (325-token prompt) | 6.8 | — |
| CPU (NEON, fallback) | 7.7 | 4.4 | — |

The NPU rows are measured on the tablet ([report, Chinese](reports/2026-11-08-y900-npu-dory.md)). Decode is bound by
memory bandwidth: two loops read about 3.5 GB per token; one loop reads a third less, at a cost in capability
(SWE-bench 59.5 → 50.3 in NVIDIA's numbers). Prompt lookup helps by task: 3–4 tokens per step when the answer copies
from the context, 1.2–1.5 when it is mostly thinking. **A static graph pays for its whole bucket every step**, so a
bigger bucket is slower; the default bucket list is 4K → 8K → 32K, and a prompt fills the small buckets first.
DSH Go X in its minimal mode (a base prompt of 4.2K tokens), measured: the first prefill takes about 12 s, every later turn computes only what is new (1–2 s), answers come at about 15 tok/s; a simple tool task took 3 minutes, a minute of it DSH's request for the session title. The standard mode's base prompt is 16K tokens, which needs the 32K bucket and is too slow.

- The NPU path is the one Qualcomm's Genie / AI Hub models use: the model is compiled on a server into static graphs
  with ExecuTorch's Qualcomm backend (16-bit activations, 4-bit weight blocks, 8-bit KV) and run on the phone by
  `libxtg_npu.so`. Tokenizer, chat template, sampling, speculative decoding, prefix cache and the HTTP API are the
  project's own C++ engine.
- The GPU (Adreno) and CPU paths remain as fallbacks; they compute on the NVFP4 weights directly.

## How

- **The two passes of the loop are unrolled with a weight copy each.** The middle 40 sublayers of Dory run twice; the
  HTP compiler turns a weight read by two ops into a DDR tensor recomputed every step, which does not fit one NPU
  context. Unrolled, the model takes about 1 GB more storage and reads the same bytes per token.
- **Four shards** (input | pass 1 | pass 2 | output), one QNN context each (a V79 context tops out near 3.5 GiB).
- **The plugin owns the KV cache.** The graphs take the caches as inputs and return only the new rows; global
  attention keeps one slot per position, sliding-window layers a fixed ring of 1088 slots that does not grow with the
  context; both masks are rebuilt every step from what each slot holds. Rewinds and the prefix-cache snapshots live in
  the plugin.
- **An 8-row decode graph**: a speculative step (prompt lookup) verifies up to 8 tokens for about the cost of one,
  since decode is bound by memory bandwidth.
- **Context buckets**: a static graph has a fixed KV length, so each bucket is its own `.pte`, and every step pays
  for the whole bucket. A session may list several; every prefill chunk runs in the smallest bucket that holds it,
  and when it does not fit the plugin moves the KV to the next bucket as it is (nothing is computed again, 1–2 s a
  move).
- **Attention reads the caches in place (4K and 8K buckets)**: the 4 q heads of a KV head are stacked along the
  rows and multiply the cache where it is: 64 ms a decode step at 4K (72 with the usual graph). At 32K the HTP
  schedules these big grouped products badly (680 ms), so the 32K bucket uses the usual graph (cache and new rows
  concatenated, K / V repeated per q head: 280 ms). The three buckets share one calibration, hence the same KV
  encodings, and a session moves between them. The HTP compiler gets the RoPE products wrong when they broadcast
  over the heads (past the first shard the cos / sin tables are graph inputs), so the in-place graphs repeat the
  tables to the full q / k shape first.

## Use

1. Install the app (a debug APK is in the Actions artifacts; the NPU needs the development build with the QNN
   libraries, see [build.md](build.md)).
2. Pick **Dory 4B · NPU** on the launcher. The tokenizer and config files download in the app; the `.pte` files and
   their `.kvmap.json` have no download source yet: export them on a server ([npu.md](npu.md)) and `adb push` them to
   the directory the launcher shows.
3. Choose the bucket list and speculative decoding, start. The server is up in seconds:
   `http://127.0.0.1:18400/v1`, model `dory`.
4. DSH Go X: Settings → Models → Add provider → Custom model API, the address above, protocol OpenAI Chat Completions,
   any API key (it may not be empty), "fetch models"; context window = the largest bucket (32768), max output 8192.
   DSH's standard mode has a base prompt of about 16K tokens; pick the minimal mode in a new session (about 4.2K),
   which is much faster.
5. In split screen or a floating window the app shows only the **rate board**: the live request's prefill / decode
   rates, the last 32 requests, memory and temperature.

## Accuracy

Per-position next-token NLL over the same 2048-token agent text: the BF16 reference 1.104, the NPU (two loops) 1.093,
level. By CPU simulation the quantization loss comes mostly from the 4-bit weights (argmax agrees with BF16 87% of the
time); 8-bit KV costs almost nothing. The export quantizes each V cache with a different encoding from the new V
written into it; the plugin maps the bytes when it writes them back (`NAME.kvmap.json`). Without that the NLL is 1.43.

License: Apache-2.0. Third-party code: see NOTICE. Model weights: NVIDIA's OpenMDW license, not in this repository;
the QNN / QAIRT libraries: Qualcomm's license, not in this repository.
