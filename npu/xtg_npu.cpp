// xtg_npu.cpp: the session behind xtg_npu.h for the shared-layout exports of tools/npu_et (Dory). Every method
// (prefill_forward with AR-P rows, kv_forward with AR-D: 1, or 8 for verify steps) reads the same caches: a full-attention instance keeps
// every position of the context (slot = position), a sliding one a ring (slot = position % ring slots). The
// graph appends the new rows itself and outputs their K / V; this file writes them back, builds both masks each
// step, rewinds, and keeps snapshots of the rings for the server's prefix cache. Caches and I/O live in RPC
// shared memory, which the NPU reads in place.
#include "xtg_npu.h"

#include <executorch/examples/qualcomm/oss_scripts/llama/runner/rpc_mem.h>
#include <executorch/extension/module/module.h>
#include <executorch/runtime/core/exec_aten/exec_aten.h>

#include <algorithm>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::aten::TensorImpl;
using executorch::extension::Module;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::MethodMeta;
using executorch::runtime::TensorInfo;

namespace {

[[noreturn]] void fail(const char* fmt, ...) __attribute__((format(printf, 1, 2)));
void fail(const char* fmt, ...) {
  char buf[512];
  va_list ap;
  va_start(ap, fmt);
  vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  throw std::runtime_error(buf);
}

#define NPU_CHECK(cond, ...)       \
  do {                             \
    if (!(cond)) fail(__VA_ARGS__); \
  } while (0)

int64_t meta_int(Module& m, const char* name, int64_t fallback) {
  if (m.method_names()->count(name) == 0) return fallback;
  auto v = m.get(name);
  NPU_CHECK(v.ok(), "%s: error %d", name, int(v.error()));
  return v->isInt() ? v->toInt() : v->toScalar().to<int64_t>();
}

double meta_double(Module& m, const char* name) {
  auto v = m.get(name);
  NPU_CHECK(v.ok(), "%s: error %d", name, int(v.error()));
  return v->isDouble() ? v->toDouble() : v->toScalar().to<double>();
}

std::unique_ptr<TensorImpl> make_impl(const TensorInfo& info, void* data) {
  return std::make_unique<TensorImpl>(info.scalar_type(), info.sizes().size(),
                                      const_cast<TensorImpl::SizesType*>(info.sizes().data()), data,
                                      const_cast<TensorImpl::DimOrderType*>(info.dim_order().data()));
}

constexpr uint16_t kSee = 65535, kHide = 0;   // the quantized 0 and -255 of the mask inputs

struct Inst {
  bool ring = false;
  int slots = 0;
  std::byte* k = nullptr;   // [heads][head_dim][slots]
  std::byte* v = nullptr;   // [heads][slots][head_dim]
};

// a method and its I/O: [tokens, mask, window mask, pos, K..., V...] -> [logits, K new..., V new...]
struct Graph {
  std::string method;
  int ar = 0;
  bool int64_tokens = false;
  std::byte *toks = nullptr, *mask = nullptr, *wmask = nullptr, *pos = nullptr, *logits = nullptr;
  int mask_cols = 0, wmask_cols = 0;
  std::vector<std::byte*> kout, vout;   // per instance: [heads][head_dim][ar], [heads][ar][head_dim]
  std::vector<std::unique_ptr<TensorImpl>> impls;
  std::vector<EValue> inputs;
  std::vector<Tensor> outputs;
};

struct Snap {
  int64_t pos = -1;
  std::vector<std::vector<std::byte>> k, v;   // the ring instances
  std::vector<int64_t> ring_pos;
};

size_t io_bytes(const MethodMeta& mm) {
  size_t b = 0;
  for (size_t i = 0; i < mm.num_inputs(); ++i) b += mm.input_tensor_meta(i)->nbytes();
  for (size_t i = 0; i < mm.num_outputs(); ++i) b += mm.output_tensor_meta(i)->nbytes();
  return b;
}

}  // namespace

struct xtg_npu {
  std::unique_ptr<Module> mod;
  std::unique_ptr<example::RpcMem> mem;
  Graph prefill, decode;
  std::vector<Inst> inst;
  int vocab = 0, heads = 0, hd = 0, ctx = 0, ring = 0, window = 0, n = 0;
  int decode_rows = 4, decode_rows_opt = 0;
  float scale = 1.f;
  int zero_point = 0;
  int64_t n_past = 0;
  std::vector<int64_t> ring_pos;   // the position each ring slot holds, -1 empty
  std::vector<Snap> snaps;
  std::vector<float> row;
  size_t cache_bytes = 0;
  std::string path;   // what open() loaded, with its settings (grow() reopens it on failure)
  bool locked = false;

  void open(const char* path, int n_snapshots, bool lock) {
    this->path = path;
    locked = lock;
    // the HTP copies the context binaries into its own memory when the methods load; locking the mapped file as
    // well (ExecuTorch's runner does) would keep a second copy of the weights resident
    mod = std::make_unique<Module>(path, lock ? Module::LoadMode::MmapUseMlockIgnoreErrors : Module::LoadMode::Mmap);
    NPU_CHECK(mod->load() == Error::Ok, "%s: cannot load", path);
    auto names = mod->method_names();
    NPU_CHECK(names.ok() && names->count("kv_forward") && names->count("prefill_forward"),
              "%s: needs the kv_forward and prefill_forward methods (a hybrid export)", path);
    NPU_CHECK(names->count("get_ring_slots"), "%s: not a shared-layout export (tools/npu_et, get_ring_slots)", path);
    for (const char* m : {"kv_forward", "prefill_forward"})
      NPU_CHECK(mod->load_method(m) == Error::Ok, "%s: cannot load method %s", path, m);
    vocab = int(meta_int(*mod, "get_vocab_size", 0));
    n = int(meta_int(*mod, "get_n_layers", 0));
    window = int(meta_int(*mod, "get_sliding_window", 0));
    ring = int(meta_int(*mod, "get_ring_slots", 0));
    ctx = int(meta_int(*mod, "get_max_context_len", 0));
    NPU_CHECK(vocab > 0 && n > 0 && ctx > 0, "%s: no vocab / layers / context", path);
    NPU_CHECK(meta_int(*mod, "get_kv_io_bit_width", 8) == 8, "%s: only 8-bit KV I/O", path);
    scale = float(meta_double(*mod, "get_logits_scale"));
    zero_point = int(meta_int(*mod, "get_logits_zero_point", 0));
    MethodMeta dm = mod->method_meta("kv_forward").get();
    MethodMeta pm = mod->method_meta("prefill_forward").get();
    decode.method = "kv_forward";
    prefill.method = "prefill_forward";
    decode.ar = int(dm.input_tensor_meta(0)->sizes()[1]);
    prefill.ar = int(pm.input_tensor_meta(0)->sizes()[1]);
    NPU_CHECK(decode.ar >= 1 && prefill.ar > decode.ar, "decode AR %d, prefill AR %d", decode.ar, prefill.ar);
    // up to this many tokens go through the decode graph (one step of decode AR rows, a verify step of a
    // speculative decoder); more take prefill steps. A 1-row decode graph runs up to 4 tokens one by one.
    decode_rows = decode_rows_opt > 0 ? decode_rows_opt : decode_rows_opt < 0 ? 0 : (decode.ar == 1 ? 4 : decode.ar);
    NPU_CHECK(int(dm.num_inputs()) == 4 + 2 * n && int(dm.num_outputs()) == 1 + 2 * n,
              "kv_forward: %zu inputs, %zu outputs for %d KV instances", dm.num_inputs(), dm.num_outputs(), n);
    // the instances from the K inputs [1, heads, head_dim, slots]
    inst.resize(size_t(n));
    for (int l = 0; l < n; ++l) {
      auto k = dm.input_tensor_meta(size_t(4 + l));
      heads = int(k->sizes()[1]);
      hd = int(k->sizes()[2]);
      const int slots = int(k->sizes()[3]);
      NPU_CHECK(slots == ctx || slots == ring, "K cache %d: %d slots (context %d, ring %d)", l, slots, ctx, ring);
      inst[size_t(l)].ring = slots == ring && ring != ctx;
      inst[size_t(l)].slots = slots;
      cache_bytes += 2 * size_t(heads) * size_t(hd) * size_t(slots);
    }
    mem = std::make_unique<example::RpcMem>(cache_bytes, io_bytes(pm), io_bytes(dm));
    for (Inst& c : inst) {
      const size_t b = size_t(heads) * size_t(hd) * size_t(c.slots);
      c.k = mem->allocate(b);
      c.v = mem->allocate(b);
      std::memset(c.k, 0, b);
      std::memset(c.v, 0, b);
    }
    bind(prefill, pm);
    bind(decode, dm);
    ring_pos.assign(size_t(ring), -1);
    snaps.resize(size_t(std::max(0, n_snapshots)));
    row.resize(size_t(vocab));
  }

  void bind(Graph& g, const MethodMeta& mm) {
    auto in = [&](size_t i, std::byte*& data) {
      auto info = mm.input_tensor_meta(i);
      data = mem->allocate(info->nbytes());
      std::memset(data, 0, info->nbytes());
      g.impls.push_back(make_impl(info.get(), data));
      g.inputs.emplace_back(Tensor(g.impls.back().get()));
      mem->add_memory_info(data, info->nbytes(), info.get());
    };
    in(0, g.toks);
    g.int64_tokens = mm.input_tensor_meta(0)->scalar_type() == ScalarType::Long;
    in(1, g.mask);
    in(2, g.wmask);
    in(3, g.pos);
    g.mask_cols = int(mm.input_tensor_meta(1)->sizes()[3]);
    g.wmask_cols = int(mm.input_tensor_meta(2)->sizes()[3]);
    NPU_CHECK(g.mask_cols == ctx + g.ar && g.wmask_cols == ring + g.ar, "%s: masks of %d / %d columns",
              g.method.c_str(), g.mask_cols, g.wmask_cols);
    for (int group = 0; group < 2; ++group)
      for (int l = 0; l < n; ++l) {
        auto info = mm.input_tensor_meta(size_t(4 + group * n + l));
        std::byte* data = group == 0 ? inst[size_t(l)].k : inst[size_t(l)].v;
        NPU_CHECK(info->nbytes() == size_t(heads) * size_t(hd) * size_t(inst[size_t(l)].slots),
                  "%s: cache %d of %zu bytes", g.method.c_str(), l, info->nbytes());
        g.impls.push_back(make_impl(info.get(), data));
        g.inputs.emplace_back(Tensor(g.impls.back().get()));
        mem->add_memory_info(data, info->nbytes(), info.get());
      }
    auto out = [&](size_t i) {
      auto info = mm.output_tensor_meta(i);
      std::byte* data = mem->allocate(info->nbytes());
      g.impls.push_back(make_impl(info.get(), data));
      g.outputs.emplace_back(Tensor(g.impls.back().get()));
      mem->add_memory_info(data, info->nbytes(), info.get());
      return data;
    };
    g.logits = out(0);
    for (int l = 0; l < n; ++l) g.kout.push_back(out(size_t(1 + l)));
    for (int l = 0; l < n; ++l) g.vout.push_back(out(size_t(1 + n + l)));
    for (size_t i = 0; i < g.outputs.size(); ++i) {
      Error e = mod->set_output(g.method, g.outputs[i], i);
      NPU_CHECK(e == Error::Ok, "%s: set_output %zu: error %d", g.method.c_str(), i, int(e));
    }
  }

  // the two masks of n rows at positions n_past ..: full caches hold positions 0 .. n_past - 1 in their slots,
  // a ring slot s holds ring_pos[s]; the rows see each other causally (and within the window for the ring)
  void fill_masks(Graph& g, int rows) {
    uint16_t* m = reinterpret_cast<uint16_t*>(g.mask);
    uint16_t* w = reinterpret_cast<uint16_t*>(g.wmask);
    std::fill(m, m + size_t(g.ar) * size_t(g.mask_cols), kHide);
    std::fill(w, w + size_t(g.ar) * size_t(g.wmask_cols), kHide);
    for (int i = 0; i < g.ar; ++i) {
      uint16_t* mr = m + size_t(i) * size_t(g.mask_cols);
      uint16_t* wr = w + size_t(i) * size_t(g.wmask_cols);
      if (i >= rows) {   // padding: itself only
        mr[ctx + i] = kSee;
        wr[ring + i] = kSee;
        continue;
      }
      const int64_t p = n_past + i;
      std::fill(mr, mr + n_past, kSee);
      for (int j = 0; j <= i; ++j) mr[ctx + j] = kSee;
      for (int s = 0; s < ring; ++s) {
        const int64_t q = ring_pos[size_t(s)];
        if (q >= 0 && q < n_past && p - q < window) wr[s] = kSee;
      }
      for (int j = std::max(0, i - window + 1); j <= i; ++j) wr[ring + j] = kSee;
    }
  }

  // the first rows of the last step's K / V into the caches
  void commit(const Graph& g, int rows) {
    const size_t c = size_t(heads) * size_t(hd);
    for (int l = 0; l < n; ++l) {
      const Inst& I = inst[size_t(l)];
      const std::byte* ko = g.kout[size_t(l)];
      const std::byte* vo = g.vout[size_t(l)];
      for (int r = 0; r < rows; ++r) {
        const size_t slot = I.ring ? size_t((n_past + r) % ring) : size_t(n_past + r);
        for (size_t e = 0; e < c; ++e) I.k[e * size_t(I.slots) + slot] = ko[e * size_t(g.ar) + size_t(r)];
        for (int h = 0; h < heads; ++h)
          std::memcpy(I.v + (size_t(h) * size_t(I.slots) + slot) * size_t(hd),
                      vo + (size_t(h) * size_t(g.ar) + size_t(r)) * size_t(hd), size_t(hd));
      }
    }
    for (int r = 0; r < rows; ++r) ring_pos[size_t((n_past + r) % ring)] = n_past + r;
  }

  void step(Graph& g, const int32_t* tokens, int rows, int want, xtg_npu_logits_fn fn, void* user, int row0) {
    NPU_CHECK(rows >= 1 && rows <= g.ar, "%s: %d tokens (AR %d)", g.method.c_str(), rows, g.ar);
    NPU_CHECK(n_past + rows <= ctx, "context full: %lld + %d tokens (%d)", (long long)n_past, rows, ctx);
    for (int i = 0; i < g.ar; ++i) {
      int32_t t = i < rows ? tokens[i] : 0;
      if (g.int64_tokens)
        reinterpret_cast<int64_t*>(g.toks)[i] = t;
      else
        reinterpret_cast<int32_t*>(g.toks)[i] = t;
      reinterpret_cast<int32_t*>(g.pos)[i] = int32_t(std::min<int64_t>(n_past + i, ctx - 1));
    }
    fill_masks(g, rows);
    auto r = mod->execute(g.method, g.inputs);
    NPU_CHECK(r.ok(), "%s: execute error %d", g.method.c_str(), int(r.error()));
    if (want != 0 && fn) {
      const uint16_t* lg = reinterpret_cast<const uint16_t*>(g.logits);
      for (int i = want == 2 ? 0 : rows - 1; i < rows; ++i) {
        const uint16_t* q = lg + size_t(i) * size_t(vocab);
        for (int j = 0; j < vocab; ++j) row[size_t(j)] = (float(q[j]) - float(zero_point)) * scale;
        fn(row0 + i, row.data(), user);
      }
    }
    commit(g, rows);
    n_past += rows;
  }

  void forward(const int32_t* tokens, int count, int want, xtg_npu_logits_fn fn, void* user) {
    for (Snap& s : snaps)   // a snapshot past this point no longer matches what the caches will hold
      if (s.pos > n_past) s.pos = -1;
    if (count <= decode_rows) {   // a decode token, a verify step, a short tail: decode-graph steps
      for (int off = 0; off < count; off += decode.ar) {
        const int m = std::min(decode.ar, count - off);
        step(decode, tokens + off, m, want == 2 ? 2 : (want == 1 && off + m == count ? 1 : 0), fn, user, off);
      }
      return;
    }
    for (int off = 0; off < count; off += prefill.ar) {
      const int m = std::min(prefill.ar, count - off);
      step(prefill, tokens + off, m, want == 2 ? 2 : (want == 1 && off + m == count ? 1 : 0), fn, user, off);
    }
  }

  bool rewind(int64_t to) {
    if (to < 0 || to > n_past) return false;
    // the ring must still hold the window before `to`
    for (int64_t q = std::max<int64_t>(0, to - window + 1); q < to; ++q)
      if (ring_pos[size_t(q % ring)] != q) return false;
    n_past = to;
    return true;
  }

  void snapshot(int slot) {
    if (slot < 0 || slot >= int(snaps.size())) return;
    Snap& s = snaps[size_t(slot)];
    s.k.clear();
    s.v.clear();
    const size_t b = size_t(heads) * size_t(hd) * size_t(ring);
    for (const Inst& I : inst)
      if (I.ring) {
        s.k.emplace_back(I.k, I.k + b);
        s.v.emplace_back(I.v, I.v + b);
      }
    s.ring_pos = ring_pos;
    s.pos = n_past;
  }

  // the session state that moves to another bucket: the first n_past positions of the full caches, the rings and
  // what their slots hold, the position, the snapshots (ring copies: the rings are the same size in every bucket)
  struct State {
    std::vector<std::vector<std::byte>> k, v;
    std::vector<int64_t> ring_pos;
    std::vector<Snap> snaps;
    int64_t n_past = 0;
  };

  State save() {
    State s;
    const size_t np = size_t(n_past);
    for (const Inst& I : inst) {
      const size_t used = I.ring ? size_t(I.slots) : np;
      std::vector<std::byte> k(size_t(heads) * size_t(hd) * used), v(k.size());
      for (size_t r = 0; r < size_t(heads) * size_t(hd); ++r)   // K rows [heads][hd] of slots
        std::memcpy(k.data() + r * used, I.k + r * size_t(I.slots), used);
      for (size_t h = 0; h < size_t(heads); ++h)   // V [heads][slots][hd]
        std::memcpy(v.data() + h * used * size_t(hd), I.v + h * size_t(I.slots) * size_t(hd), used * size_t(hd));
      s.k.push_back(std::move(k));
      s.v.push_back(std::move(v));
    }
    s.ring_pos = ring_pos;
    s.snaps = std::move(snaps);
    s.n_past = n_past;
    return s;
  }

  void load(State& s) {
    NPU_CHECK(s.k.size() == inst.size(), "%zu KV instances, the session had %zu", inst.size(), s.k.size());
    NPU_CHECK(s.n_past <= ctx && s.ring_pos.size() == size_t(ring), "context %d for %lld positions, ring %d", ctx,
              (long long)s.n_past, ring);
    for (size_t l = 0; l < inst.size(); ++l) {
      const Inst& I = inst[l];
      const size_t used = s.k[l].size() / (size_t(heads) * size_t(hd));
      NPU_CHECK(used <= size_t(I.slots) && (!I.ring || used == size_t(I.slots)), "instance %zu: %zu of %d slots", l,
                used, I.slots);
      for (size_t r = 0; r < size_t(heads) * size_t(hd); ++r)
        std::memcpy(I.k + r * size_t(I.slots), s.k[l].data() + r * used, used);
      for (size_t h = 0; h < size_t(heads); ++h)
        std::memcpy(I.v + h * size_t(I.slots) * size_t(hd), s.v[l].data() + h * used * size_t(hd), used * size_t(hd));
    }
    ring_pos = s.ring_pos;
    n_past = s.n_past;
    for (size_t i = 0; i < snaps.size() && i < s.snaps.size(); ++i)   // a smaller bucket drops what it cannot hold
      if (s.snaps[i].pos <= ctx) snaps[i] = std::move(s.snaps[i]);
  }

  bool restore(int slot) {
    if (slot < 0 || slot >= int(snaps.size()) || snaps[size_t(slot)].pos < 0) return false;
    const Snap& s = snaps[size_t(slot)];
    size_t i = 0;
    for (Inst& I : inst)
      if (I.ring) {
        std::memcpy(I.k, s.k[i].data(), s.k[i].size());
        std::memcpy(I.v, s.v[i].data(), s.v[i].size());
        ++i;
      }
    ring_pos = s.ring_pos;
    n_past = s.pos;
    return true;
  }
};

namespace {
void set_err(char* err, int len, const char* what) {
  if (err && len > 0) snprintf(err, size_t(len), "%s", what);
}

int option_int(const char* options, const char* key, int fallback) {
  if (!options) return fallback;
  std::string o = options, k = std::string("\"") + key + "\"";
  size_t p = o.find(k);
  if (p == std::string::npos) return fallback;
  p = o.find(':', p + k.size());
  return p == std::string::npos ? fallback : atoi(o.c_str() + p + 1);
}
}  // namespace

extern "C" {

xtg_npu* xtg_npu_open(const char* pte_path, const char* options, char* err, int err_len) {
  auto h = std::make_unique<xtg_npu>();
  try {
    h->decode_rows_opt = option_int(options, "decode_rows", 0);
    h->open(pte_path, option_int(options, "snapshots", 2), option_int(options, "mlock", 0) != 0);
  } catch (const std::exception& e) {
    set_err(err, err_len, e.what());
    return nullptr;
  }
  return h.release();
}

int xtg_npu_info(xtg_npu* h, char* out, int out_len) {
  int rings = 0;
  for (const Inst& I : h->inst) rings += I.ring;
  int len = snprintf(out, size_t(out_len),
                     "{\"vocab\":%d,\"context\":%d,\"max_position\":%d,\"prefill_ar\":%d,\"decode_ar\":%d,"
                     "\"kv_instances\":%d,\"ring_instances\":%d,\"ring_slots\":%d,\"sliding_window\":%d,"
                     "\"kv_bytes\":%zu,\"snapshots\":%zu,\"logits_scale\":%.9g,\"logits_zero_point\":%d}",
                     h->vocab, h->ctx, h->ctx - 1, h->prefill.ar, h->decode.ar, h->n, rings, h->ring, h->window,
                     h->cache_bytes, h->snaps.size(), double(h->scale), h->zero_point);
  return len < out_len ? len : -1;
}

int xtg_npu_forward(xtg_npu* h, const int32_t* tokens, int n, int want, xtg_npu_logits_fn fn, void* user, char* err,
                    int err_len) {
  try {
    h->forward(tokens, n, want, fn, user);
  } catch (const std::exception& e) {
    set_err(err, err_len, e.what());
    return -1;
  }
  return 0;
}

int64_t xtg_npu_pos(xtg_npu* h) { return h->n_past; }

int xtg_npu_rewind(xtg_npu* h, int64_t n) { return h->rewind(n) ? 0 : -1; }

int xtg_npu_snapshot_slots(xtg_npu* h) { return int(h->snaps.size()); }

int64_t xtg_npu_snapshot_pos(xtg_npu* h, int slot) {
  return slot >= 0 && slot < int(h->snaps.size()) ? h->snaps[size_t(slot)].pos : -1;
}

void xtg_npu_snapshot(xtg_npu* h, int slot) { h->snapshot(slot); }

int xtg_npu_restore(xtg_npu* h, int slot) { return h->restore(slot) ? 0 : -1; }

int xtg_npu_grow(xtg_npu* h, const char* pte_path, char* err, int err_len) {
  const int n_snapshots = int(h->snaps.size()), rows = h->decode_rows_opt;
  const bool lock = h->locked;
  const std::string old = h->path;
  xtg_npu::State s = h->save();
  auto reopen = [&](const std::string& p) {
    auto next = std::make_unique<xtg_npu>();
    next->decode_rows_opt = rows;
    next->open(p.c_str(), n_snapshots, lock);
    next->load(s);
    return next;
  };
  // the contexts go first (the module, then the shared memory it reads): two buckets' weights may not fit together
  h->mod.reset();
  h->mem.reset();
  try {
    *h = std::move(*reopen(pte_path));
    return 0;
  } catch (const std::exception& e) {
    set_err(err, err_len, e.what());
  }
  try {
    *h = std::move(*reopen(old));
  } catch (const std::exception& e) {   // the session is gone: forward and the rest fail until it is closed
    set_err(err, err_len, (std::string(err ? err : "") + "; reopening " + old + ": " + e.what()).c_str());
  }
  return -1;
}

void xtg_npu_close(xtg_npu* h) { delete h; }

}  // extern "C"
