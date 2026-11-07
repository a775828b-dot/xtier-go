// KV cache of one attention layer instance and the attention kernel that reads it.
//
// A full-attention cache holds every position up to its capacity; a sliding-window cache is a ring of the last
// `slots` positions. Memory is reserved as address space (MAP_NORESERVE) and only becomes resident as positions
// are written, so a 256K capacity costs nothing until the context grows. FP8 rows keep one FP32 scale per
// (position, head): Dory's keys are unit vectors per head (entries near 1/16), which a fixed E4M3 grid resolves
// poorly in its subnormal range.
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "xtg/threadpool.h"

namespace xtg {

enum class KvDtype { BF16, FP8 };
const char* kv_dtype_name(KvDtype d);

class KvCache {
 public:
  // ring = 0: linear cache of `slots` positions; ring > 0: ring of `ring` slots (slots is ignored).
  KvCache(int kv_heads, int head_dim, int64_t slots, int ring, KvDtype dtype);
  ~KvCache();
  KvCache(const KvCache&) = delete;
  KvCache& operator=(const KvCache&) = delete;

  bool ring() const { return ring_; }
  int64_t slots() const { return slots_; }
  KvDtype dtype() const { return dtype_; }

  // Rows of positions [pos, pos + n) from FP32 k, v laid out [n][kv_heads][head_dim]. A ring keeps the last
  // `slots` of them.
  void store(int64_t pos, int n, const float* k, const float* v);
  // FP32 K and V rows of head h for positions [p0, p1), all resident (a ring: within its last `slots`).
  void load(int h, int64_t p0, int64_t p1, float* k, float* v) const;

  size_t bytes_per_position() const;          // all heads, K and V, with scales
  size_t reserved_bytes() const { return bytes_; }
  // Resident estimate after `n_past` positions.
  size_t used_bytes(int64_t n_past) const;

  // The whole ring as bytes (prefix-cache snapshots) and back.
  void save(std::vector<uint8_t>& out) const;
  void restore(const std::vector<uint8_t>& in);
  // Give the pages back (the next writes start from zero pages).
  void release();

 private:
  int kv_heads_, head_dim_;
  int64_t slots_;
  bool ring_;
  KvDtype dtype_;
  size_t elem_ = 2;
  uint8_t* base_ = nullptr;
  size_t bytes_ = 0;
  uint8_t* k_ = nullptr;
  uint8_t* v_ = nullptr;
  float* ks_ = nullptr;   // FP8: [slots][kv_heads]
  float* vs_ = nullptr;
};

// Causal attention of this call's n_q queries (positions pos0 .. pos0 + n_q - 1) over the cached positions before
// pos0 and this call's own keys. window > 0: a query at t sees keys t - window + 1 .. t (vLLM's sliding window).
// The cache is read only; the caller stores this call's K / V afterwards.
struct AttnArgs {
  int n_q = 0;
  int64_t pos0 = 0;
  int heads = 0, kv_heads = 0, head_dim = 0;
  int window = 0;
  float scale = 1.f;
  const float* q = nullptr;     // [n_q][heads][head_dim]
  const float* k = nullptr;     // [n_q][kv_heads][head_dim]
  const float* v = nullptr;
  const KvCache* cache = nullptr;
  float* out = nullptr;         // [n_q][heads][head_dim]
};
void attention(ThreadPool& pool, const AttnArgs& a);

}  // namespace xtg
