// NVIDIA Nemotron-Labs Dory: a looped (latent-recurrent) normalized GPT, ported from vLLM main
// (vllm/model_executor/models/dory.py, PR #60149).
//
// 72 physical sublayers, each attention-only or MLP-only, in three groups: input (run once), recurrent (run
// `loops` times, the input re-injected each time as justnorm(state + input)) and output (run once). Every
// sublayer ends in nGPT's slerp residual; there is no final norm. Attention is GQA with nGPT q/k norms after
// RoPE, an output gate and a norm over all heads; sliding-window and full-attention layers interleave.
//
// KV modes (vLLM's recurrent_kv_cache_mode): per_loop keeps a cache per loop for the recurrent attention layers;
// last_loop keeps one, so earlier tokens are seen with the last loop's keys (less memory, slightly different
// predictions).
#pragma once

#include <memory>
#include <string>
#include <vector>

#include "xtg/json.h"
#include "xtg/model.h"

namespace xtg::dory {


void register_pack();

// The end-of-sequence ids: generation_config.json, else config.json.
std::vector<int> read_eos(const std::string& dir, const json& config);

// backend=npu (npu.cpp): the .pte of the QNN export through libxtg_npu, and its memory estimate.
std::unique_ptr<Model> load_npu(const std::string& dir, const json& config, const json& options);
json estimate_npu(const std::string& dir, const json& config, const json& options);

// The options this pack takes (keys of a launch request; the card's choices are a subset):
//   loops          1..4 (default: the checkpoint's n_recurrent_loops)
//   kv_mode        "per_loop" | "last_loop" (default: the checkpoint's)
//   kv_dtype       "fp8" (default) | "bf16" (accuracy checks on the server)
//   context        KV capacity in tokens (<= max_position_embeddings)
//   prefill_chunk  tokens per forward chunk (last_loop results depend on it, as in vLLM)
//   snapshots      sliding-window snapshots kept for the prefix cache
//   backend        "cpu" (default) | "gpu" | "gpu-linear" | "npu"
//   npu_model      backend=npu: the .pte (default: the only one in the checkpoint directory)
//   npu_plugin     backend=npu: the plugin library (default libxtg_npu.so, found like any shared library)
json option_schema();

}  // namespace xtg::dory
