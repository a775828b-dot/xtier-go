#include "dory.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

#include "xtg/common.h"
#include "xtg/fp.h"
#include "xtg/kernels.h"
#include "xtg/kv.h"
#include "xtg/model.h"
#include "xtg/opencl.h"
#include "xtg/safetensors.h"

namespace xtg::dory {
namespace {



enum Kind { kMlp = 0, kSliding = 1, kFull = 2 };

// sliding-window rings keep this many positions beyond the window, so a session can rewind up to kRewindSlack + 1
// positions (speculative decoding drops the rejected drafts) without a snapshot
constexpr int kRewindSlack = 16;

struct Config {
  int hidden = 0, ffn = 0, heads = 0, kv_heads = 0, head_dim = 0, vocab = 0, layers = 0;
  int n_in = 0, n_rec = 0, n_out = 0, window = 0, default_loops = 1;
  int64_t max_pos = 0;
  float theta[3] = {0, 0, 0};
  float init_range = 0, alpha_init = 0, gate_init = 0;
  std::string default_kv_mode;
  std::vector<int> kind;   // per physical layer

  bool recurrent(int layer) const { return layer >= n_in && layer < n_in + n_rec; }
};

int64_t get_int(const json& o, const char* key, int64_t def) {
  if (!o.contains(key) || o[key].is_null()) return def;
  const json& v = o[key];
  if (v.is_number_integer()) return v.get<int64_t>();
  if (v.is_string()) {
    const std::string& s = v.get_ref<const std::string&>();
    char* end = nullptr;
    long long n = std::strtoll(s.c_str(), &end, 10);
    XTG_CHECK(end && *end == 0 && !s.empty(), "option %s: \"%s\" is not an integer", key, s.c_str());
    return n;
  }
  fail("option %s: not an integer", key);
}

std::string get_str(const json& o, const char* key, const std::string& def) {
  if (!o.contains(key) || o[key].is_null()) return def;
  XTG_CHECK(o[key].is_string(), "option %s: not a string", key);
  return o[key].get<std::string>();
}

Config parse_config(const json& c, const std::string& dir) {
  auto need = [&](const char* key) -> const json& {
    XTG_CHECK(c.contains(key) && !c[key].is_null(), "%s: config.json has no %s", dir.c_str(), key);
    return c[key];
  };
  Config k;
  XTG_CHECK(c.value("model_type", "") == "dory", "%s: not a Dory checkpoint", dir.c_str());
  k.hidden = need("hidden_size").get<int>();
  k.ffn = need("intermediate_size").get<int>();
  k.heads = need("num_attention_heads").get<int>();
  k.kv_heads = need("num_key_value_heads").get<int>();
  k.head_dim = need("head_dim").get<int>();
  k.vocab = need("vocab_size").get<int>();
  k.layers = need("num_hidden_layers").get<int>();
  k.n_in = need("n_input_layers").get<int>();
  k.n_rec = need("n_recurrent_layers").get<int>();
  k.n_out = need("n_output_layers").get<int>();
  k.window = c.value("sliding_window", 0);
  k.default_loops = c.value("n_recurrent_loops", 1);
  k.default_kv_mode = c.value("recurrent_kv_cache_mode", std::string("per_loop"));
  k.max_pos = need("max_position_embeddings").get<int64_t>();
  k.init_range = need("initializer_range").get<float>();
  k.alpha_init = need("ngpt_alpha_init").get<float>();
  const json& g0 = c.contains("attention_output_gate_logit_scale_init") ? c["attention_output_gate_logit_scale_init"] : json();
  k.gate_init = g0.is_null() ? std::sqrt(float(k.hidden)) : g0.get<float>();
  XTG_CHECK(k.n_in + k.n_rec + k.n_out == k.layers, "%s: input + recurrent + output layers != %d", dir.c_str(), k.layers);
  XTG_CHECK(k.heads % k.kv_heads == 0 && k.head_dim % 2 == 0, "%s: unsupported attention shape", dir.c_str());
  XTG_CHECK(!c.value("attention_bias", false) && !c.value("mlp_bias", false), "%s: Dory has no biases", dir.c_str());
  XTG_CHECK(c.value("hidden_act", "silu") == "silu" && !c.value("tie_word_embeddings", false),
            "%s: Dory uses SiLU and untied embeddings", dir.c_str());
  for (const auto& t : need("layer_types")) {
    std::string s = t.get<std::string>();
    if (s == "mlp")
      k.kind.push_back(kMlp);
    else if (s == "sliding_attention")
      k.kind.push_back(kSliding);
    else if (s == "full_attention")
      k.kind.push_back(kFull);
    else
      fail("%s: unknown layer type %s", dir.c_str(), s.c_str());
  }
  XTG_CHECK(int(k.kind.size()) == k.layers, "%s: layer_types has %zu entries", dir.c_str(), k.kind.size());
  const json& rope = need("rope_parameters");
  for (auto [kind, key] : {std::pair{kSliding, "sliding_attention"}, std::pair{kFull, "full_attention"}}) {
    if (!rope.contains(key)) continue;
    const json& r = rope[key];
    XTG_CHECK(r.value("rope_type", "default") == "default" && r.value("partial_rotary_factor", 1.0) == 1.0,
              "%s: only default full-width RoPE is supported", dir.c_str());
    k.theta[kind] = r.at("rope_theta").get<float>();
  }
  for (int t : k.kind) XTG_CHECK(t == kMlp || k.theta[t] > 0, "%s: a layer type has no rope_theta", dir.c_str());
  XTG_CHECK(k.window > 0 || std::count(k.kind.begin(), k.kind.end(), kSliding) == 0, "%s: sliding layers without a window",
            dir.c_str());
  const json q = c.value("quantization_config", json());
  XTG_CHECK(q.value("quant_algo", "") == "W4A16_NVFP4", "%s: only the W4A16_NVFP4 (ModelOpt) checkpoint is supported",
            dir.c_str());
  return k;
}

struct Options {
  bool gpu = false;        // an OpenCL device holds the weights
  bool resident = false;   // ... and the whole step (gpu); else only the linear layers (gpu-linear)
  int loops = 1;
  bool per_loop = true;
  KvDtype kv_dtype = KvDtype::FP8;
  int64_t context = 32768;
  int chunk = 512;
  int snapshots = 2;
};

Options parse_options(const Config& c, const json& o) {
  Options p;
  const auto backend = get_str(o, "backend", "cpu");
  XTG_CHECK(backend == "cpu" || backend == "gpu" || backend == "gpu-linear", "backend: %s (cpu | gpu | gpu-linear)",
            backend.c_str());
  p.gpu = backend != "cpu";
  p.resident = backend == "gpu";
  p.loops = int(get_int(o, "loops", c.default_loops));
  XTG_CHECK(p.loops >= 1 && p.loops <= 4, "loops: %d (1..4)", p.loops);
  std::string mode = get_str(o, "kv_mode", c.default_kv_mode);
  XTG_CHECK(mode == "per_loop" || mode == "last_loop", "kv_mode: %s (per_loop | last_loop)", mode.c_str());
  p.per_loop = mode == "per_loop";
  std::string dt = get_str(o, "kv_dtype", "fp8");
  XTG_CHECK(dt == "bf16" || dt == "fp8", "kv_dtype: %s (bf16 | fp8)", dt.c_str());
  p.kv_dtype = dt == "fp8" ? KvDtype::FP8 : KvDtype::BF16;
  p.context = get_int(o, "context", 32768);
  XTG_CHECK(p.context >= 256 && p.context <= c.max_pos, "context: %lld (256..%lld)", (long long)p.context,
            (long long)c.max_pos);
  p.chunk = int(get_int(o, "prefill_chunk", 512));
  XTG_CHECK(p.chunk >= 16 && p.chunk <= 8192, "prefill_chunk: %d (16..8192)", p.chunk);
  p.snapshots = int(get_int(o, "snapshots", 2));
  XTG_CHECK(p.snapshots >= 0 && p.snapshots <= 8, "snapshots: %d (0..8)", p.snapshots);
  return p;
}

// KV bytes of one position of one layer instance
size_t kv_row_bytes(const Config& c, KvDtype d) {
  return 2 * size_t(c.kv_heads) * (size_t(c.head_dim) * (d == KvDtype::FP8 ? 1 : 2) + (d == KvDtype::FP8 ? 4 : 0));
}

json estimate(const Config& c, const Options& p, size_t weight_bytes) {
  size_t full = 0, ring = 0;
  int full_inst = 0, ring_inst = 0;
  for (int l = 0; l < c.layers; ++l) {
    if (c.kind[size_t(l)] == kMlp) continue;
    int inst = c.recurrent(l) && p.per_loop ? p.loops : 1;
    if (c.kind[size_t(l)] == kFull) {
      full_inst += inst;
      full += size_t(inst) * size_t(p.context) * kv_row_bytes(c, p.kv_dtype);
    } else {
      ring_inst += inst;
      ring += size_t(inst) * size_t(c.window + kRewindSlack) * kv_row_bytes(c, p.kv_dtype);
    }
  }
  size_t q = size_t(c.heads) * size_t(c.head_dim), kv = size_t(c.kv_heads) * size_t(c.head_dim);
  size_t work = size_t(p.chunk) * (4 * size_t(c.hidden) + 3 * q + 2 * kv + 2 * size_t(c.ffn) + size_t(c.hidden)) * 4 +
                16 * size_t(c.vocab) * 4;
  size_t embed = size_t(c.vocab) * size_t(c.hidden) * 2;
  size_t snaps = size_t(p.snapshots) * ring;
  // the embedding table is mapped but only the rows of seen tokens become resident
  // Initial GPU backend copies packed weights; count both copies conservatively.
  size_t gpu_weights = p.gpu ? weight_bytes - embed : 0;
  size_t gpu_work = p.gpu ? size_t(p.chunk) * (size_t(c.ffn) + 2 * std::max(q + kv, size_t(c.ffn))) * 4 +
                              16 * size_t(c.vocab) * 4 : 0;
  size_t resident = weight_bytes - embed + full + ring + snaps + work + gpu_weights + gpu_work;
  return json{{"weights_bytes", weight_bytes},          {"embedding_bytes", embed},
              {"kv_full_bytes", full},                  {"kv_window_bytes", ring},
              {"kv_full_instances", full_inst},          {"kv_window_instances", ring_inst},
              {"kv_bytes_per_token", size_t(full_inst) * kv_row_bytes(c, p.kv_dtype)},
              {"snapshot_bytes", snaps},                {"work_bytes", work},
              {"gpu_weight_bytes", gpu_weights}, {"gpu_work_bytes", gpu_work}, {"resident_bytes", resident}};
}

size_t weights_file_bytes(const std::string& dir) {
  std::ifstream f(dir + "/model.safetensors", std::ios::binary | std::ios::ate);
  XTG_CHECK(f.good(), "%s: no model.safetensors", dir.c_str());
  return size_t(f.tellg());
}

std::vector<int> read_eos(const std::string& dir, const json& config) {
  std::vector<int> ids;
  auto add = [&](const json& v) {
    if (v.is_number_integer())
      ids.push_back(v.get<int>());
    else if (v.is_array())
      for (const auto& x : v) ids.push_back(x.get<int>());
  };
  std::ifstream g(dir + "/generation_config.json");
  if (g.good()) {
    json gc = json::parse(g);
    if (gc.contains("eos_token_id")) add(gc["eos_token_id"]);
  }
  if (ids.empty() && config.contains("eos_token_id")) add(config["eos_token_id"]);
  return ids;
}

struct Layer {
  int kind = kMlp;
  bool recurrent = false;
  int attn = -1;   // index among the attention layers (the session's caches)
  Nvfp4Weight q, k, v, g, o, gate, up, down;
  std::vector<float> lr, qscale, gscale, sg, su;
  float theta = 0.f;
  int window = 0;
};

class DoryModel;

class DorySession : public Session {
 public:
  explicit DorySession(DoryModel& m);
  ~DorySession() override {
    if (std::getenv("XTG_GPU_PROFILE")) log("decode attention + KV store: %.6fs", decode_attention_s_);
  }
  int64_t n_past() const override { return n_; }
  void forward(const int32_t* tokens, int n, Logits want, const LogitsFn& on_logits) override;
  bool rewind(int64_t n) override;
  void snapshot(int slot) override;
  bool restore(int slot) override;
  int64_t snapshot_pos(int slot) const override {
    return slot >= 0 && slot < int(snaps_.size()) ? snaps_[size_t(slot)].pos : -1;
  }
  int snapshot_slots() const override { return int(snaps_.size()); }
  void reset() override;
  size_t kv_bytes() const override;

 private:
  void run_chunk(const int32_t* tokens, int m, int64_t pos0);
  void sublayer(const Layer& L, float* x, int m, int64_t pos0, int inst, bool store);
  void drop_snapshots_above(int64_t pos);

  DoryModel& M_;
  int64_t n_ = 0;
  double decode_attention_s_ = 0;
  std::vector<std::vector<std::unique_ptr<KvCache>>> caches_;   // [attention layer][instance]
  struct Snap {
    int64_t pos = -1;
    std::vector<std::vector<uint8_t>> rings;
  };
  std::vector<Snap> snaps_;
  std::vector<float> x_, input_, state_, hq_, hk_, hv_, hg_, ha_, ho_, mg_, mu_, logits_;
};

// The same forward with everything on the OpenCL device (backend=gpu): activations and the KV caches stay in
// device buffers, every operation is a queued kernel, and the host only uploads the embeddings / RoPE tables of a
// chunk and reads back the logits it asked for. FP32 activations and accumulation, the CPU session's formulas.
class GpuDorySession : public Session {
 public:
  explicit GpuDorySession(DoryModel& m);
  ~GpuDorySession() override;
  int64_t n_past() const override { return n_; }
  void forward(const int32_t* tokens, int n, Logits want, const LogitsFn& on_logits) override;
  bool rewind(int64_t n) override;
  void snapshot(int slot) override;
  bool restore(int slot) override;
  int64_t snapshot_pos(int slot) const override {
    return slot >= 0 && slot < int(snaps_.size()) ? snaps_[size_t(slot)].pos : -1;
  }
  int snapshot_slots() const override { return int(snaps_.size()); }
  void reset() override;
  size_t kv_bytes() const override;

 private:
  using Mem = OpenClBackend::Mem;
  struct Cache {   // FP8 rows [slots][kv_heads][head_dim], one scale per (slot, head)
    Mem k = nullptr, v = nullptr, ks = nullptr, vs = nullptr;
    int64_t slots = 0;
    bool ring = false;
  };
  struct Vecs {   // a layer's vectors on the device
    Mem lr = nullptr, qscale = nullptr, gscale = nullptr, sg = nullptr, su = nullptr;
  };
  void run_chunk(const int32_t* tokens, int m, int64_t pos0);
  void sublayer(size_t li, int m, int64_t pos0, int inst, bool store);
  Mem upload(const std::vector<float>& v);
  size_t cache_bytes(const Cache& c) const;
  void drop_snapshots_above(int64_t pos);

  DoryModel& M_;
  OpenClBackend& G_;
  int64_t n_ = 0;
  std::vector<std::vector<Cache>> caches_;   // [attention layer][instance]
  std::vector<Vecs> vecs_;                   // per physical layer
  struct Snap {
    int64_t pos = -1;
    std::vector<Cache> rings;
  };
  std::vector<Snap> snaps_;
  Mem x_ = nullptr, input_ = nullptr, state_ = nullptr, proj_ = nullptr, att_ = nullptr, gu_ = nullptr, o_ = nullptr;
  Mem rope_[3] = {nullptr, nullptr, nullptr}, hx_ = nullptr, logits_dev_ = nullptr;
  Mem part_ = nullptr;   // decode attention partials [kv head][split][group head][max, sum, d outputs]
  static constexpr int kMaxSplits = 64;
  std::vector<float> host_, logits_;
};

class DoryModel : public Model {
 public:
  DoryModel(const std::string& dir, const json& config, const json& options, ThreadPool& pool)
      : c(parse_config(config, dir)), o(parse_options(c, options)), pool(pool) {
    double t0 = now_s();
    st = std::make_unique<SafeTensors>(dir + "/model.safetensors");
    st->advise_willneed();
    const int H = c.hidden, F = c.ffn, Q = c.heads * c.head_dim, KV = c.kv_heads * c.head_dim;
    embed = reinterpret_cast<const uint16_t*>(st->get("backbone.embeddings.weight", "BF16", {c.vocab, H}).data);
    lm_head = {reinterpret_cast<const uint16_t*>(st->get("lm_head.weight", "BF16", {c.vocab, H}).data), c.vocab, H};
    logit_scale = bf16_vec("lm_head.logit_scale", c.vocab);
    for (float& s : logit_scale) s /= c.init_range;
    const float alpha_scale = c.alpha_init / c.init_range;
    const float sqk_factor = std::sqrt(float(c.head_dim)) / c.init_range;
    const float gate_factor = std::sqrt(c.gate_init) / c.init_range;
    const float suv_factor = std::sqrt(float(H));
    layers.resize(size_t(c.layers));
    int n_attn = 0;
    for (int i = 0; i < c.layers; ++i) {
      Layer& L = layers[size_t(i)];
      std::string p = "backbone.layers." + std::to_string(i) + ".";
      L.kind = c.kind[size_t(i)];
      L.recurrent = c.recurrent(i);
      if (L.kind == kMlp) {
        L.gate = nvfp4(p + "mixer.gate_proj", F, H);
        L.up = nvfp4(p + "mixer.up_proj", F, H);
        L.down = nvfp4(p + "mixer.down_proj", H, F);
        L.sg = bf16_vec(p + "mixer.suv_gate", F);
        L.su = bf16_vec(p + "mixer.suv_up", F);
        for (float& s : L.sg) s *= suv_factor;
        for (float& s : L.su) s *= suv_factor;
        L.lr = bf16_vec(p + "mlp_alpha", H);
      } else {
        L.attn = n_attn++;
        L.q = nvfp4(p + "mixer.q_proj", Q, H);
        L.k = nvfp4(p + "mixer.k_proj", KV, H);
        L.v = nvfp4(p + "mixer.v_proj", KV, H);
        L.g = nvfp4(p + "mixer.g_proj", Q, H);
        L.o = nvfp4(p + "mixer.o_proj", H, Q);
        L.qscale = bf16_vec(p + "mixer.sqk", Q);
        L.gscale = bf16_vec(p + "mixer.s_gate", Q);
        for (float& s : L.qscale) s = (s * sqk_factor) * (s * sqk_factor);
        for (float& s : L.gscale) s = (s * gate_factor) * (s * gate_factor);
        L.lr = bf16_vec(p + "attn_alpha", H);
        L.theta = c.theta[L.kind];
        L.window = L.kind == kSliding ? c.window : 0;
      }
      for (float& s : L.lr) s = std::fabs(s * alpha_scale);
    }
    if (o.gpu) {
      gpu = std::make_unique<OpenClBackend>();
      for (const auto& L : layers)
        for (auto w : {&L.q, &L.k, &L.v, &L.g, &L.o, &L.gate, &L.up, &L.down}) gpu->prepare(*w);
      gpu->prepare(lm_head);
      log("OpenCL packed weights: %zu bytes", gpu->weight_bytes());
    }
    info_.model_type = "dory";
    info_.vocab = c.vocab;
    info_.context = o.context;
    info_.eos_ids = read_eos(dir, config);
    json mem = estimate(c, o, st->file_size());
    info_.describe = json{{"model_type", "dory"},
                          {"layers", {{"input", c.n_in}, {"recurrent", c.n_rec}, {"output", c.n_out}}},
                          {"loops", o.loops},
                          {"kv_mode", o.per_loop ? "per_loop" : "last_loop"},
                          {"kv_dtype", kv_dtype_name(o.kv_dtype)},
                          {"context", o.context},
                          {"prefill_chunk", o.chunk},
                          {"snapshots", o.snapshots},
                          {"kernels", o.resident ? "opencl" : o.gpu ? "opencl-linear+neon" : kernel_set_name()},
                          {"gpu_device", gpu ? gpu->device_name() : ""},
                          {"threads", pool.size()},
                          {"memory", mem}};
    log("dory: %d layers (%d / %d x %d / %d), %s, %s KV, context %lld, weights mapped in %.2f s", c.layers, c.n_in,
        c.n_rec, o.loops, c.n_out, o.per_loop ? "per_loop" : "last_loop", kv_dtype_name(o.kv_dtype),
        (long long)o.context, now_s() - t0);
  }

  const ModelInfo& info() const override { return info_; }
  std::unique_ptr<Session> new_session() override;
  void matmul(const std::vector<const Nvfp4Weight*>& ws, const float* x, int m, const std::vector<float*>& ys) {
    if (gpu) gpu->matmul(ws, x, m, ys);
    else matmul_nvfp4(pool, ws, x, m, ys);
  }
  void head(const float* x, int m, float* y) {
    if (gpu) gpu->matmul(lm_head, x, m, y);
    else matmul_bf16(pool, lm_head, x, m, y);
  }

  Config c;
  Options o;
  ThreadPool& pool;
  std::unique_ptr<SafeTensors> st;
  std::vector<Layer> layers;
  const uint16_t* embed = nullptr;
  Bf16Weight lm_head;
  std::vector<float> logit_scale;
  ModelInfo info_;
  std::unique_ptr<OpenClBackend> gpu;

 private:
  Nvfp4Weight nvfp4(const std::string& name, int n_out, int k) {
    const TensorView& w = st->get(name + ".weight", "U8", {n_out, k / 2});
    const TensorView& s = st->get(name + ".weight_scale", "F8_E4M3", {n_out, k / 16});
    const TensorView& g = st->get(name + ".weight_scale_2", "F32", {});
    Nvfp4Weight r;
    r.codes = w.data;
    r.scales = s.data;
    std::memcpy(&r.global, g.data, 4);
    r.n_out = n_out;
    r.k = k;
    return r;
  }
  std::vector<float> bf16_vec(const std::string& name, int n) {
    const TensorView& t = st->get(name, "BF16", {n});
    std::vector<float> v(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
      uint16_t h;
      std::memcpy(&h, t.data + 2 * i, 2);
      v[size_t(i)] = bf16_to_f32(h);
    }
    return v;
  }
};

DorySession::DorySession(DoryModel& m) : M_(m) {
  const Config& c = m.c;
  for (const Layer& L : m.layers) {
    if (L.kind == kMlp) continue;
    int inst = L.recurrent && m.o.per_loop ? m.o.loops : 1;
    caches_.emplace_back();
    for (int i = 0; i < inst; ++i)
      caches_.back().push_back(std::make_unique<KvCache>(c.kv_heads, c.head_dim, m.o.context,
                                                         L.kind == kSliding ? c.window + kRewindSlack : 0, m.o.kv_dtype));
  }
  snaps_.resize(size_t(m.o.snapshots));
  const size_t ch = size_t(m.o.chunk), H = size_t(c.hidden), Q = size_t(c.heads) * size_t(c.head_dim),
               KV = size_t(c.kv_heads) * size_t(c.head_dim), F = size_t(c.ffn);
  x_.resize(ch * H);
  input_.resize(ch * H);
  state_.resize(ch * H);
  hq_.resize(ch * Q);
  hk_.resize(ch * KV);
  hv_.resize(ch * KV);
  hg_.resize(ch * Q);
  ha_.resize(ch * Q);
  ho_.resize(ch * H);
  mg_.resize(ch * F);
  mu_.resize(ch * F);
}

void DorySession::sublayer(const Layer& L, float* x, int m, int64_t pos0, int inst, bool store) {
  check_cancelled();
  const Config& c = M_.c;
  ThreadPool& pool = M_.pool;
  const int H = c.hidden;
  if (L.kind == kMlp) {
    M_.matmul({&L.gate, &L.up}, x, m, {mg_.data(), mu_.data()});
    swiglu_scaled(mg_.data(), mu_.data(), L.sg.data(), L.su.data(), m, c.ffn);
    justnorm(mg_.data(), m, c.ffn);
    M_.matmul({&L.down}, mg_.data(), m, {ho_.data()});
    slerp_residual(x, ho_.data(), L.lr.data(), m, H);
    return;
  }
  const int Q = c.heads * c.head_dim;
  M_.matmul({&L.q, &L.k, &L.v, &L.g}, x, m, {hq_.data(), hk_.data(), hv_.data(), hg_.data()});
  rope(hq_.data(), m, c.heads, c.head_dim, pos0, L.theta);
  rope(hk_.data(), m, c.kv_heads, c.head_dim, pos0, L.theta);
  // nGPT qk-norm after RoPE: q per head normalized and scaled by (sqk * f)^2, k per head normalized
  justnorm(hq_.data(), m * c.heads, c.head_dim);
  for (int r = 0; r < m; ++r) {
    float* qr = hq_.data() + size_t(r) * size_t(Q);
    for (int j = 0; j < Q; ++j) qr[j] *= L.qscale[size_t(j)];
  }
  justnorm(hk_.data(), m * c.kv_heads, c.head_dim);
  KvCache& cache = *caches_[size_t(L.attn)][size_t(inst)];
  AttnArgs a;
  a.n_q = m;
  a.pos0 = pos0;
  a.heads = c.heads;
  a.kv_heads = c.kv_heads;
  a.head_dim = c.head_dim;
  a.window = L.window;
  a.scale = 1.f / std::sqrt(float(c.head_dim));
  a.q = hq_.data();
  a.k = hk_.data();
  a.v = hv_.data();
  a.cache = &cache;
  a.out = ha_.data();
  const double attention_started = m == 1 ? now_s() : 0;
  attention(pool, a);
  if (store) cache.store(pos0, m, hk_.data(), hv_.data());
  if (m == 1) decode_attention_s_ += now_s() - attention_started;
  // output gate (sigmoid, before o_proj), then a norm over all heads
  for (int r = 0; r < m; ++r) {
    float* ar = ha_.data() + size_t(r) * size_t(Q);
    const float* gr = hg_.data() + size_t(r) * size_t(Q);
    for (int j = 0; j < Q; ++j) ar[j] *= 1.f / (1.f + std::exp(-gr[j] * L.gscale[size_t(j)]));
  }
  justnorm(ha_.data(), m, Q);
  M_.matmul({&L.o}, ha_.data(), m, {ho_.data()});
  slerp_residual(x, ho_.data(), L.lr.data(), m, H);
}

void DorySession::run_chunk(const int32_t* tokens, int m, int64_t pos0) {
  const Config& c = M_.c;
  const int H = c.hidden;
  const size_t n = size_t(m) * size_t(H);
  for (int r = 0; r < m; ++r) {
    XTG_CHECK(tokens[r] >= 0 && tokens[r] < c.vocab, "token %d out of range", tokens[r]);
    bf16_row(M_.embed, tokens[r], H, x_.data() + size_t(r) * size_t(H));
  }
  int l = 0;
  for (; l < c.n_in; ++l) sublayer(M_.layers[size_t(l)], x_.data(), m, pos0, 0, true);
  std::memcpy(input_.data(), x_.data(), n * sizeof(float));
  std::memcpy(state_.data(), x_.data(), n * sizeof(float));
  for (int i = 0; i < M_.o.loops; ++i) {
    for (size_t j = 0; j < n; ++j) x_[j] = state_[j] + input_[j];
    justnorm(x_.data(), m, H);
    // last_loop: every loop attends to this chunk's own keys, the cache keeps the last loop's
    bool store = M_.o.per_loop || i == M_.o.loops - 1;
    for (int r = c.n_in; r < c.n_in + c.n_rec; ++r)
      sublayer(M_.layers[size_t(r)], x_.data(), m, pos0, M_.o.per_loop ? i : 0, store);
    std::memcpy(state_.data(), x_.data(), n * sizeof(float));
    justnorm(state_.data(), m, H);
  }
  std::memcpy(x_.data(), state_.data(), n * sizeof(float));
  for (l = c.n_in + c.n_rec; l < c.layers; ++l) sublayer(M_.layers[size_t(l)], x_.data(), m, pos0, 0, true);
}

void DorySession::drop_snapshots_above(int64_t pos) {
  for (Snap& s : snaps_)
    if (s.pos > pos) {
      s.pos = -1;
      s.rings.clear();
    }
}

void DorySession::forward(const int32_t* tokens, int n, Logits want, const LogitsFn& on_logits) {
  const Config& c = M_.c;
  XTG_CHECK(n_ + n <= M_.o.context, "context full: %lld + %d > %lld tokens", (long long)n_, n, (long long)M_.o.context);
  drop_snapshots_above(n_);
  const int rows_per_head = 16;
  for (int off = 0; off < n; off += M_.o.chunk) {
    check_cancelled();
    int m = std::min(M_.o.chunk, n - off);
    run_chunk(tokens + off, m, n_);
    n_ += m;
    int r0 = want == Logits::All ? 0 : (want == Logits::Last && off + m == n ? m - 1 : m);
    for (int r = r0; r < m; r += rows_per_head) {
      check_cancelled();
      int rows = std::min(rows_per_head, m - r);
      logits_.resize(size_t(rows) * size_t(c.vocab));
      M_.head(x_.data() + size_t(r) * size_t(c.hidden), rows, logits_.data());
      for (int i = 0; i < rows; ++i) {
        float* lg = logits_.data() + size_t(i) * size_t(c.vocab);
        for (int j = 0; j < c.vocab; ++j) lg[j] *= M_.logit_scale[size_t(j)];
        on_logits(off + r + i, lg);
      }
    }
  }
}

bool DorySession::rewind(int64_t n) {
  if (n == n_) return true;
  if (n > n_ || n < 0) return false;
  // a ring of window + slack slots still holds what position n needs when it lost at most slack + 1 positions
  bool rings = M_.c.window > 0 && std::count(M_.c.kind.begin(), M_.c.kind.end(), int(kSliding)) > 0;
  if (rings && n_ > M_.c.window + kRewindSlack && n < n_ - kRewindSlack - 1) return false;
  n_ = n;
  drop_snapshots_above(n);
  return true;
}

void DorySession::snapshot(int slot) {
  if (slot < 0 || slot >= int(snaps_.size())) return;
  Snap& s = snaps_[size_t(slot)];
  s.rings.clear();
  for (auto& layer : caches_)
    for (auto& cache : layer)
      if (cache->ring()) {
        s.rings.emplace_back();
        cache->save(s.rings.back());
      }
  s.pos = n_;
}

bool DorySession::restore(int slot) {
  if (slot < 0 || slot >= int(snaps_.size()) || snaps_[size_t(slot)].pos < 0) return false;
  Snap& s = snaps_[size_t(slot)];
  size_t i = 0;
  for (auto& layer : caches_)
    for (auto& cache : layer)
      if (cache->ring()) cache->restore(s.rings[i++]);
  n_ = s.pos;
  drop_snapshots_above(n_);
  return true;
}

void DorySession::reset() {
  n_ = 0;
  for (Snap& s : snaps_) {
    s.pos = -1;
    s.rings.clear();
  }
  for (auto& layer : caches_)
    for (auto& cache : layer) cache->release();
}

size_t DorySession::kv_bytes() const {
  size_t b = 0;
  for (const auto& layer : caches_)
    for (const auto& cache : layer) b += cache->used_bytes(n_);
  for (const Snap& s : snaps_)
    for (const auto& r : s.rings) b += r.size();
  return b;
}

// --- GPU-resident session ---

std::unique_ptr<Session> DoryModel::new_session() {
  if (o.resident) return std::make_unique<GpuDorySession>(*this);
  return std::make_unique<DorySession>(*this);
}

// Row kernels run one work-group per row (or per row and head); every norm is justnorm: x / sqrt(sum x^2 + 1e-6).
const char* kDoryCl = R"CLC(
float wg_sum(float v, __local float* red) {
  const int t = get_local_id(0), n = get_local_size(0);
  red[t] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = n / 2; s > 0; s >>= 1) {
    if (t < s) red[t] += red[t + s];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float r = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}
float wg_max(float v, __local float* red) {
  const int t = get_local_id(0), n = get_local_size(0);
  red[t] = v;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int s = n / 2; s > 0; s >>= 1) {
    if (t < s) red[t] = fmax(red[t], red[t + s]);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float r = red[0];
  barrier(CLK_LOCAL_MEM_FENCE);
  return r;
}

// out = justnorm(a + b) (add) or justnorm(a); out may be a
__kernel void add_norm(__global float* a, __global const float* b, __global float* out, int w, int add) {
  __local float red[256];
  const int t = get_local_id(0);
  const size_t r = get_group_id(0) * (size_t)w;
  float s = 0;
  for (int j = t; j < w; j += 256) { float v = a[r + j] + (add ? b[r + j] : 0.f); s += v * v; }
  const float f = 1.f / sqrt(wg_sum(s, red) + 1e-6f);
  for (int j = t; j < w; j += 256) out[r + j] = (a[r + j] + (add ? b[r + j] : 0.f)) * f;
}

// x = justnorm(norm(x) + lr * (norm(o) - norm(x)))
__kernel void slerp(__global float* x, __global const float* o, __global const float* lr, int w) {
  __local float red[256];
  const int t = get_local_id(0);
  const size_t r = get_group_id(0) * (size_t)w;
  float sa = 0, sb = 0;
  for (int j = t; j < w; j += 256) { sa += x[r + j] * x[r + j]; sb += o[r + j] * o[r + j]; }
  const float fa = 1.f / sqrt(wg_sum(sa, red) + 1e-6f), fb = 1.f / sqrt(wg_sum(sb, red) + 1e-6f);
  float s = 0;
  for (int j = t; j < w; j += 256) {
    float ya = x[r + j] * fa, yb = o[r + j] * fb, v = ya + lr[j] * (yb - ya);
    x[r + j] = v;
    s += v * v;
  }
  const float f = 1.f / sqrt(wg_sum(s, red) + 1e-6f);
  for (int j = t; j < w; j += 256) x[r + j] *= f;
}

// gate block of gu (rows of w) = justnorm(silu(g * sg) * (u * su)), u at u_off
__kernel void swiglu_norm(__global float* gu, int u_off, __global const float* sg, __global const float* su, int w) {
  __local float red[256];
  const int t = get_local_id(0);
  const size_t r = get_group_id(0) * (size_t)w;
  float s = 0;
  for (int j = t; j < w; j += 256) {
    float z = gu[r + j] * sg[j], v = z / (1.f + exp(-z)) * (gu[u_off + r + j] * su[j]);
    gu[r + j] = v;
    s += v * v;
  }
  const float f = 1.f / sqrt(wg_sum(s, red) + 1e-6f);
  for (int j = t; j < w; j += 256) gu[r + j] *= f;
}

// att = justnorm(att * sigmoid(g * gs)), g at g_off of proj
__kernel void gate_norm(__global float* att, __global const float* proj, int g_off, __global const float* gs, int w) {
  __local float red[256];
  const int t = get_local_id(0);
  const size_t r = get_group_id(0) * (size_t)w;
  float s = 0;
  for (int j = t; j < w; j += 256) {
    float v = att[r + j] * (1.f / (1.f + exp(-proj[g_off + r + j] * gs[j])));
    att[r + j] = v;
    s += v * v;
  }
  const float f = 1.f / sqrt(wg_sum(s, red) + 1e-6f);
  for (int j = t; j < w; j += 256) att[r + j] *= f;
}

// NeoX RoPE (cos / sin table of the chunk: [m][d/2] each), then justnorm per head, then times scale (q)
__kernel void rope_norm(__global float* x, int x_off, __global const float* cs, int m, int heads, int d,
                        __global const float* scale, int use_scale) {
  __local float red[128];
  const int g = get_group_id(0), r = g / heads, h = g % heads, i = get_local_id(0), hh = d / 2;
  __global float* v = x + x_off + ((size_t)r * heads + h) * d;
  const float c = cs[(size_t)r * hh + i], s = cs[(size_t)(m + r) * hh + i];
  float x1 = v[i], x2 = v[i + hh];
  float o1 = x1 * c - x2 * s, o2 = x2 * c + x1 * s;
  const float f = 1.f / sqrt(wg_sum(o1 * o1 + o2 * o2, red) + 1e-6f);
  o1 *= f;
  o2 *= f;
  if (use_scale) {
    o1 *= scale[h * d + i];
    o2 *= scale[h * d + i + hh];
  }
  v[i] = o1;
  v[i + hh] = o2;
}

float e4m3(uchar b) {
  uint e = (b >> 3) & 15, m = b & 7;
  float v = e ? as_float(((e + 120) << 23) | (m << 20)) : (float)m * 0.001953125f;
  return (b & 128) ? -v : v;
}
// FP32 -> E4M3, the CPU KvCache's rounding (nearest even, saturating to 448)
uchar e4m3_bits(float x) {
  uint bits = as_uint(x), sign = (bits >> 24) & 0x80, a = bits & 0x7fffffff;
  if (a > 0x7f800000) return (uchar)(sign | 0x7f);
  if (a < 0x3c800000) {
    float t = as_float(a) * 512.f, lo = floor(t), frac = t - lo;
    uint steps = (uint)lo;
    steps += (frac > 0.5f) || (frac == 0.5f && (steps & 1));
    return (uchar)(sign | steps);
  }
  uint r = a + 0x7ffff + ((a >> 20) & 1);
  return (uchar)(sign | min(((((r >> 23) - 120) << 3) | ((r >> 20) & 7)), 0x7eu));
}

// this call's K / V rows (k at k_off, v at v_off of proj) into the cache, one work-group of d per (row, KV head)
__kernel void kv_store(__global const float* proj, int k_off, int v_off, __global uchar* ck, __global uchar* cv,
                       __global float* cks, __global float* cvs, long pos0, int n, int kv_heads, int d, long slots, int ring) {
  __local float red[256];
  const int g = get_group_id(0), r = g / kv_heads, h = g % kv_heads, j = get_local_id(0);
  const size_t src = ((size_t)r * kv_heads + h) * d + j;
  const float kx = proj[k_off + src], vx = proj[v_off + src];
  const float ak = wg_max(fabs(kx), red), av = wg_max(fabs(vx), red);
  if (ring && r < n - slots) return;   // a ring keeps the last `slots` positions of a long chunk
  const long p = pos0 + r, slot = ring ? p % slots : p;
  const size_t idx = (size_t)slot * kv_heads + h;
  const float sk = ak > 0.f ? ak / 448.f : 1.f, sv = av > 0.f ? av / 448.f : 1.f;
  ck[idx * d + j] = e4m3_bits(kx * (1.f / sk));
  cv[idx * d + j] = e4m3_bits(vx * (1.f / sv));
  if (j == 0) {
    cks[idx] = sk;
    cvs[idx] = sv;
  }
}

// softmax(q k / sqrt(d)) v for one (row, head) per work-group of 256 (= d): keys before pos0 from the FP8 cache,
// this call's keys from proj (FP32), causal and windowed; tiles of 256 keys with an online softmax
__kernel void attend(__global const float* proj, int k_off, int v_off, __global const uchar* ck, __global const uchar* cv,
                     __global const float* cks, __global const float* cvs, __global float* out, int heads, int kv_heads,
                     int d, long pos0, int window, long slots, int ring, float sm) {
  __local float q[256], p[256], red[256];
  const int g = get_group_id(0), r = g / heads, h = g % heads, kvh = h / (heads / kv_heads), t = get_local_id(0);
  const long pos = pos0 + r, lo = window > 0 ? max(0L, pos - window + 1) : 0L;
  q[t] = proj[((size_t)r * heads + h) * d + t];
  barrier(CLK_LOCAL_MEM_FENCE);
  float m = -INFINITY, l = 0.f, acc = 0.f;
  for (long k0 = lo; k0 <= pos; k0 += 256) {
    const long key = k0 + t;
    float s = -INFINITY;
    if (key <= pos) {
      float dot = 0.f;
      if (key < pos0) {
        const size_t idx = (size_t)(ring ? key % slots : key) * kv_heads + kvh;
        __global const uchar* kr = ck + idx * d;
        for (int j = 0; j < d; j++) dot += q[j] * e4m3(kr[j]);
        dot *= cks[idx];
      } else {
        __global const float* kr = proj + k_off + ((size_t)(key - pos0) * kv_heads + kvh) * d;
        for (int j = 0; j < d; j++) dot += q[j] * kr[j];
      }
      s = dot * sm;
    }
    const float mx = wg_max(s, red), m_new = fmax(m, mx), corr = m == -INFINITY ? 0.f : exp(m - m_new);
    const float e = key <= pos ? exp(s - m_new) : 0.f;
    p[t] = e;
    l = l * corr + wg_sum(e, red);
    acc *= corr;
    const int nk = (int)min(256L, pos - k0 + 1);
    for (int j = 0; j < nk; j++) {
      const long kk = k0 + j;
      float v;
      if (kk < pos0) {
        const size_t idx = (size_t)(ring ? kk % slots : kk) * kv_heads + kvh;
        v = e4m3(cv[idx * d + t]) * cvs[idx];
      } else {
        v = proj[v_off + ((size_t)(kk - pos0) * kv_heads + kvh) * d + t];
      }
      acc += p[j] * v;
    }
    m = m_new;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  out[((size_t)r * heads + h) * d + t] = acc / l;
}

// Decode (one query row) as flash-decoding: a work-group per (KV head, split of the visible keys) for the GROUP
// q-heads sharing that KV head; FP8 K tiles staged in local memory, V read once per key for all GROUP heads;
// partial (max, sum, output) per q-head into part, attend_merge combines the splits. d = 256, 256 work-items.
#define GROUP 4
#define TILE 64
__kernel void attend_split(__global const float* proj, int k_off, int v_off, __global const uchar* ck,
                           __global const uchar* cv, __global const float* cks, __global const float* cvs,
                           __global float* part, int kv_heads, int d, long pos0, int window, long slots, int ring,
                           float sm, int splits) {
  __local float q[GROUP][256], sc[GROUP][TILE], red[256], tmax[GROUP], tsum[GROUP];
  __local uchar kt[TILE][256];
  __local float ks[TILE];
  const int g = get_group_id(0), kvh = g / splits, sp = g % splits, t = get_local_id(0);
  const int hi = t / TILE, j = t % TILE;   // the (head, key) a work-item scores
  const long lo = window > 0 ? max(0L, pos0 - window + 1) : 0L, span = pos0 - lo + 1;
  const long s0 = lo + span * sp / splits, s1 = lo + span * (sp + 1) / splits;
  for (int i = 0; i < GROUP; i++) q[i][t] = proj[(size_t)(kvh * GROUP + i) * d + t];
  float m[GROUP], l[GROUP], acc[GROUP];
  for (int i = 0; i < GROUP; i++) { m[i] = -INFINITY; l[i] = 0.f; acc[i] = 0.f; }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (long k0 = s0; k0 < s1; k0 += TILE) {
    const int nk = (int)min((long)TILE, s1 - k0);
    // stage the tile's keys: cached ones as FP8 rows, the current token (pos0, only in the last split) from proj
    for (int e = t; e < TILE * 256 / 16; e += 256) {
      const int kj = e / 16, part16 = (e % 16) * 16;
      const long key = k0 + kj;
      if (kj < nk && key < pos0) {
        const size_t idx = (size_t)(ring ? key % slots : key) * kv_heads + kvh;
        vstore16(vload16(0, ck + idx * d + part16), 0, &kt[kj][part16]);
        if (part16 == 0) ks[kj] = cks[idx];
      }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float s = -INFINITY;
    if (j < nk) {
      const long key = k0 + j;
      float dot = 0.f;
      if (key < pos0) {
        for (int e = 0; e < 256; e++) dot += q[hi][e] * e4m3(kt[j][e]);
        dot *= ks[j];
      } else {
        __global const float* kr = proj + k_off + (size_t)kvh * d;
        for (int e = 0; e < 256; e++) dot += q[hi][e] * kr[e];
      }
      s = dot * sm;
    }
    // per head: the tile's max and sum over its TILE work-items
    red[t] = s;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = TILE / 2; st > 0; st >>= 1) {
      if (j < st) red[t] = fmax(red[t], red[t + st]);
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j == 0) tmax[hi] = red[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    const float m_new = fmax(m[hi], tmax[hi]);
    const float p = j < nk ? exp(s - m_new) : 0.f;
    sc[hi][j] = p;
    red[t] = p;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int st = TILE / 2; st > 0; st >>= 1) {
      if (j < st) red[t] += red[t + st];
      barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j == 0) tsum[hi] = red[t];
    barrier(CLK_LOCAL_MEM_FENCE);
    float corr[GROUP];
    for (int i = 0; i < GROUP; i++) {
      const float mn = fmax(m[i], tmax[i]);
      corr[i] = m[i] == -INFINITY ? 0.f : exp(m[i] - mn);
      l[i] = l[i] * corr[i] + tsum[i];
      acc[i] *= corr[i];
      m[i] = mn;
    }
    // work-item t = output dimension t: every key's V once, for all GROUP heads
    for (int kj = 0; kj < nk; kj++) {
      const long key = k0 + kj;
      float v;
      if (key < pos0) {
        const size_t idx = (size_t)(ring ? key % slots : key) * kv_heads + kvh;
        v = e4m3(cv[idx * d + t]) * cvs[idx];
      } else {
        v = proj[v_off + (size_t)kvh * d + t];
      }
      for (int i = 0; i < GROUP; i++) acc[i] += sc[i][kj] * v;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  for (int i = 0; i < GROUP; i++) {
    __global float* o = part + ((size_t)(kvh * splits + sp) * GROUP + i) * (2 + d);
    if (t == 0) { o[0] = m[i]; o[1] = l[i]; }
    o[2 + t] = acc[i];
  }
}

// one work-group per q-head: the splits' partial softmax outputs into the attention output row
__kernel void attend_merge(__global const float* part, __global float* out, int d, int splits) {
  const int h = get_group_id(0), kvh = h / GROUP, i = h % GROUP, t = get_local_id(0);
  float M = -INFINITY;
  for (int s = 0; s < splits; s++) M = fmax(M, part[((size_t)(kvh * splits + s) * GROUP + i) * (2 + d)]);
  float L = 0.f, acc = 0.f;
  for (int s = 0; s < splits; s++) {
    __global const float* p = part + ((size_t)(kvh * splits + s) * GROUP + i) * (2 + d);
    if (p[0] == -INFINITY) continue;
    const float w = exp(p[0] - M);
    L += p[1] * w;
    acc += p[2 + t] * w;
  }
  out[(size_t)h * d + t] = acc / L;
}
)CLC";

GpuDorySession::GpuDorySession(DoryModel& m) : M_(m), G_(*m.gpu) {
  const Config& c = m.c;
  XTG_CHECK(m.o.kv_dtype == KvDtype::FP8, "backend=gpu keeps an FP8 KV cache (kv_dtype=fp8)");
  XTG_CHECK(c.head_dim == 256 && c.hidden % 256 == 0 && c.heads == 4 * c.kv_heads,
            "backend=gpu: kernels written for head_dim 256 and 4 query heads per KV head");
  const size_t ch = size_t(m.o.chunk), H = size_t(c.hidden), Q = size_t(c.heads) * size_t(c.head_dim),
               KV = size_t(c.kv_heads) * size_t(c.head_dim), F = size_t(c.ffn), half = size_t(c.head_dim) / 2;
  for (const Layer& L : m.layers) {
    Vecs v;
    v.lr = upload(L.lr);
    if (L.kind == kMlp) {
      v.sg = upload(L.sg);
      v.su = upload(L.su);
    } else {
      v.qscale = upload(L.qscale);
      v.gscale = upload(L.gscale);
      int inst = L.recurrent && m.o.per_loop ? m.o.loops : 1;
      caches_.emplace_back();
      for (int i = 0; i < inst; ++i) {
        Cache k;
        k.ring = L.kind == kSliding;
        k.slots = k.ring ? c.window + kRewindSlack : m.o.context;
        k.k = G_.alloc(size_t(k.slots) * KV);
        k.v = G_.alloc(size_t(k.slots) * KV);
        k.ks = G_.alloc(size_t(k.slots) * size_t(c.kv_heads) * 4);
        k.vs = G_.alloc(size_t(k.slots) * size_t(c.kv_heads) * 4);
        caches_.back().push_back(k);
      }
    }
    vecs_.push_back(v);
  }
  snaps_.resize(size_t(m.o.snapshots));
  x_ = G_.alloc(ch * H * 4);
  input_ = G_.alloc(ch * H * 4);
  state_ = G_.alloc(ch * H * 4);
  proj_ = G_.alloc(ch * (2 * Q + 2 * KV) * 4);
  att_ = G_.alloc(ch * Q * 4);
  gu_ = G_.alloc(ch * 2 * F * 4);
  o_ = G_.alloc(ch * H * 4);
  for (int kind : {int(kSliding), int(kFull)}) rope_[kind] = G_.alloc(ch * 2 * half * 4);
  hx_ = G_.alloc(16 * H * 4);
  logits_dev_ = G_.alloc(16 * size_t(c.vocab) * 4);
  part_ = G_.alloc(size_t(c.kv_heads) * kMaxSplits * size_t(c.heads / c.kv_heads) * (2 + size_t(c.head_dim)) * 4);
}

GpuDorySession::~GpuDorySession() {
  G_.finish();
  auto drop = [&](Cache& k) {
    for (Mem b : {k.k, k.v, k.ks, k.vs}) G_.release(b);
  };
  for (auto& layer : caches_)
    for (auto& k : layer) drop(k);
  for (auto& s : snaps_)
    for (auto& k : s.rings) drop(k);
  for (auto& v : vecs_)
    for (Mem b : {v.lr, v.qscale, v.gscale, v.sg, v.su}) G_.release(b);
  for (Mem b : {x_, input_, state_, proj_, att_, gu_, o_, rope_[kSliding], rope_[kFull], hx_, logits_dev_, part_}) G_.release(b);
}

OpenClBackend::Mem GpuDorySession::upload(const std::vector<float>& v) {
  Mem b = G_.alloc(v.size() * 4);
  G_.write(b, 0, v.data(), v.size() * 4);
  return b;
}

void GpuDorySession::sublayer(size_t li, int m, int64_t pos0, int inst, bool store) {
  check_cancelled();
  const Config& c = M_.c;
  const Layer& L = M_.layers[li];
  const Vecs& V = vecs_[li];
  const int H = c.hidden, F = c.ffn, Q = c.heads * c.head_dim, KV = c.kv_heads * c.head_dim, d = c.head_dim;
  auto slerp = G_.kernel(kDoryCl, "", "slerp");
  if (L.kind == kMlp) {
    G_.matmul({&L.gate, &L.up}, x_, m, gu_, 0);
    G_.launch(G_.kernel(kDoryCl, "", "swiglu_norm"), {size_t(m) * 256, 1}, {256, 1}, gu_, m * F, V.sg, V.su, F);
    G_.matmul({&L.down}, gu_, m, o_, 0);
    G_.launch(slerp, {size_t(m) * 256, 1}, {256, 1}, x_, o_, V.lr, H);
    return;
  }
  const int k_off = m * Q, v_off = m * (Q + KV), g_off = m * (Q + 2 * KV);
  G_.matmul({&L.q, &L.k, &L.v, &L.g}, x_, m, proj_, 0);
  auto rope = G_.kernel(kDoryCl, "", "rope_norm");
  Mem none = nullptr;
  G_.launch(rope, {size_t(m) * c.heads * (d / 2), 1}, {size_t(d / 2), 1}, proj_, 0, rope_[L.kind], m, c.heads, d, V.qscale, 1);
  G_.launch(rope, {size_t(m) * c.kv_heads * (d / 2), 1}, {size_t(d / 2), 1}, proj_, k_off, rope_[L.kind], m, c.kv_heads, d,
            none, 0);
  const Cache& K = caches_[size_t(L.attn)][size_t(inst)];
  const int64_t p0 = pos0, slots = K.slots;   // OpenCL long
  const int ring = K.ring ? 1 : 0, window = L.window;
  const float sm = 1.f / std::sqrt(float(d));
  if (m == 1) {   // decode: splits of up to 256 visible keys, at most kMaxSplits
    const int64_t span = window > 0 ? std::min<int64_t>(pos0 + 1, window) : pos0 + 1;
    const int splits = int(std::clamp<int64_t>((span + 255) / 256, 1, kMaxSplits));
    G_.launch(G_.kernel(kDoryCl, "", "attend_split"), {size_t(c.kv_heads) * size_t(splits) * 256, 1}, {256, 1}, proj_, k_off,
              v_off, K.k, K.v, K.ks, K.vs, part_, c.kv_heads, d, p0, window, slots, ring, sm, splits);
    G_.launch(G_.kernel(kDoryCl, "", "attend_merge"), {size_t(c.heads) * 256, 1}, {256, 1}, part_, att_, d, splits);
  } else {
    G_.launch(G_.kernel(kDoryCl, "", "attend"), {size_t(m) * c.heads * 256, 1}, {256, 1}, proj_, k_off, v_off, K.k, K.v,
              K.ks, K.vs, att_, c.heads, c.kv_heads, d, p0, window, slots, ring, sm);
  }
  if (store)
    G_.launch(G_.kernel(kDoryCl, "", "kv_store"), {size_t(m) * c.kv_heads * size_t(d), 1}, {size_t(d), 1}, proj_, k_off,
              v_off, K.k, K.v, K.ks, K.vs, p0, m, c.kv_heads, d, slots, ring);
  G_.launch(G_.kernel(kDoryCl, "", "gate_norm"), {size_t(m) * 256, 1}, {256, 1}, att_, proj_, g_off, V.gscale, Q);
  G_.matmul({&L.o}, att_, m, o_, 0);
  G_.launch(slerp, {size_t(m) * 256, 1}, {256, 1}, x_, o_, V.lr, H);
}

void GpuDorySession::run_chunk(const int32_t* tokens, int m, int64_t pos0) {
  const Config& c = M_.c;
  const int H = c.hidden, half = c.head_dim / 2;
  host_.resize(size_t(m) * size_t(H));
  for (int r = 0; r < m; ++r) {
    XTG_CHECK(tokens[r] >= 0 && tokens[r] < c.vocab, "token %d out of range", tokens[r]);
    bf16_row(M_.embed, tokens[r], H, host_.data() + size_t(r) * size_t(H));
  }
  G_.write(x_, 0, host_.data(), host_.size() * 4);
  // the RoPE tables of this chunk, as the CPU computes them (float positions times 1 / theta^(2i/d))
  for (int kind : {int(kSliding), int(kFull)}) {
    std::vector<float> cs(size_t(2) * size_t(m) * size_t(half));
    for (int i = 0; i < half; ++i) {
      float inv = 1.f / std::pow(c.theta[kind], float(2 * i) / float(c.head_dim));
      for (int r = 0; r < m; ++r) {
        float a = float(pos0 + r) * inv;
        cs[size_t(r) * size_t(half) + size_t(i)] = std::cos(a);
        cs[size_t(m + r) * size_t(half) + size_t(i)] = std::sin(a);
      }
    }
    G_.write(rope_[kind], 0, cs.data(), cs.size() * 4);
  }
  const size_t bytes = size_t(m) * size_t(H) * 4;
  size_t l = 0;
  for (; l < size_t(c.n_in); ++l) sublayer(l, m, pos0, 0, true);
  G_.copy(x_, 0, input_, 0, bytes);
  G_.copy(x_, 0, state_, 0, bytes);
  for (int i = 0; i < M_.o.loops; ++i) {
    G_.launch(G_.kernel(kDoryCl, "", "add_norm"), {size_t(m) * 256, 1}, {256, 1}, state_, input_, x_, H, 1);
    bool store = M_.o.per_loop || i == M_.o.loops - 1;
    for (size_t r = size_t(c.n_in); r < size_t(c.n_in + c.n_rec); ++r) sublayer(r, m, pos0, M_.o.per_loop ? i : 0, store);
    G_.launch(G_.kernel(kDoryCl, "", "add_norm"), {size_t(m) * 256, 1}, {256, 1}, x_, x_, state_, H, 0);
  }
  G_.copy(state_, 0, x_, 0, bytes);
  for (l = size_t(c.n_in + c.n_rec); l < size_t(c.layers); ++l) sublayer(l, m, pos0, 0, true);
}

void GpuDorySession::forward(const int32_t* tokens, int n, Logits want, const LogitsFn& on_logits) {
  const Config& c = M_.c;
  XTG_CHECK(n_ + n <= M_.o.context, "context full: %lld + %d > %lld tokens", (long long)n_, n, (long long)M_.o.context);
  drop_snapshots_above(n_);
  const int rows_per_head = 16;
  const size_t H = size_t(c.hidden);
  for (int off = 0; off < n; off += M_.o.chunk) {
    check_cancelled();
    int m = std::min(M_.o.chunk, n - off);
    run_chunk(tokens + off, m, n_);
    n_ += m;
    int r0 = want == Logits::All ? 0 : (want == Logits::Last && off + m == n ? m - 1 : m);
    for (int r = r0; r < m; r += rows_per_head) {
      int rows = std::min(rows_per_head, m - r);
      G_.copy(x_, size_t(r) * H * 4, hx_, 0, size_t(rows) * H * 4);
      G_.matmul(M_.lm_head, hx_, rows, logits_dev_);
      logits_.resize(size_t(rows) * size_t(c.vocab));
      G_.read(logits_dev_, 0, logits_.data(), logits_.size() * 4);
      for (int i = 0; i < rows; ++i) {
        float* lg = logits_.data() + size_t(i) * size_t(c.vocab);
        for (int j = 0; j < c.vocab; ++j) lg[j] *= M_.logit_scale[size_t(j)];
        on_logits(off + r + i, lg);
      }
    }
  }
  G_.finish();
}

void GpuDorySession::drop_snapshots_above(int64_t pos) {
  for (Snap& s : snaps_)
    if (s.pos > pos) s.pos = -1;
}

bool GpuDorySession::rewind(int64_t n) {
  if (n == n_) return true;
  if (n > n_ || n < 0) return false;
  bool rings = M_.c.window > 0 && std::count(M_.c.kind.begin(), M_.c.kind.end(), int(kSliding)) > 0;
  if (rings && n_ > M_.c.window + kRewindSlack && n < n_ - kRewindSlack - 1) return false;
  n_ = n;
  drop_snapshots_above(n);
  return true;
}

void GpuDorySession::snapshot(int slot) {
  if (slot < 0 || slot >= int(snaps_.size())) return;
  Snap& s = snaps_[size_t(slot)];
  size_t i = 0;
  for (auto& layer : caches_)
    for (auto& k : layer) {
      if (!k.ring) continue;
      const size_t rows = size_t(k.slots) * size_t(M_.c.kv_heads);
      if (i == s.rings.size()) {   // first use of this slot: device buffers like the ring's
        Cache c = k;
        c.k = G_.alloc(rows * size_t(M_.c.head_dim));
        c.v = G_.alloc(rows * size_t(M_.c.head_dim));
        c.ks = G_.alloc(rows * 4);
        c.vs = G_.alloc(rows * 4);
        s.rings.push_back(c);
      }
      const Cache& d = s.rings[i++];
      G_.copy(k.k, 0, d.k, 0, rows * size_t(M_.c.head_dim));
      G_.copy(k.v, 0, d.v, 0, rows * size_t(M_.c.head_dim));
      G_.copy(k.ks, 0, d.ks, 0, rows * 4);
      G_.copy(k.vs, 0, d.vs, 0, rows * 4);
    }
  s.pos = n_;
}

bool GpuDorySession::restore(int slot) {
  if (slot < 0 || slot >= int(snaps_.size()) || snaps_[size_t(slot)].pos < 0) return false;
  Snap& s = snaps_[size_t(slot)];
  size_t i = 0;
  for (auto& layer : caches_)
    for (auto& k : layer) {
      if (!k.ring) continue;
      const Cache& d = s.rings[i++];
      const size_t rows = size_t(k.slots) * size_t(M_.c.kv_heads);
      G_.copy(d.k, 0, k.k, 0, rows * size_t(M_.c.head_dim));
      G_.copy(d.v, 0, k.v, 0, rows * size_t(M_.c.head_dim));
      G_.copy(d.ks, 0, k.ks, 0, rows * 4);
      G_.copy(d.vs, 0, k.vs, 0, rows * 4);
    }
  n_ = s.pos;
  drop_snapshots_above(n_);
  return true;
}

void GpuDorySession::reset() {
  n_ = 0;
  for (Snap& s : snaps_) s.pos = -1;
}

size_t GpuDorySession::cache_bytes(const Cache& c) const {
  return 2 * size_t(c.slots) * size_t(M_.c.kv_heads) * (size_t(M_.c.head_dim) + 4);
}

size_t GpuDorySession::kv_bytes() const {
  size_t b = 0;
  for (const auto& layer : caches_)
    for (const auto& k : layer) b += cache_bytes(k);
  for (const Snap& s : snaps_)
    for (const auto& k : s.rings) b += cache_bytes(k);
  return b;
}

}  // namespace

json option_schema() {
  return json{{"loops", "1..4"},          {"kv_mode", {"per_loop", "last_loop"}}, {"kv_dtype", {"bf16", "fp8"}},
              {"context", "256..max_position_embeddings"}, {"prefill_chunk", "16..8192"}, {"snapshots", "0..8"}};
}

void register_pack() {
  ModelPack p;
  p.load = [](const std::string& dir, const json& config, const json& options, ThreadPool& pool) {
    return std::unique_ptr<Model>(new DoryModel(dir, config, options, pool));
  };
  p.estimate = [](const std::string& dir, const json& config, const json& options) {
    Config c = parse_config(config, dir);
    return estimate(c, parse_options(c, options), weights_file_bytes(dir));
  };
  xtg::register_pack("dory", std::move(p));
}

}  // namespace xtg::dory
