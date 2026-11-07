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
  bool gpu = false;
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
  XTG_CHECK(backend == "cpu" || backend == "gpu", "backend: %s (cpu | gpu)", backend.c_str());
  p.gpu = backend == "gpu";
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
      ring += size_t(inst) * size_t(c.window) * kv_row_bytes(c, p.kv_dtype);
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
                          {"kernels", o.gpu ? "opencl-linear+neon" : kernel_set_name()},
                          {"gpu_device", gpu ? gpu->device_name() : ""},
                          {"threads", pool.size()},
                          {"memory", mem}};
    log("dory: %d layers (%d / %d x %d / %d), %s, %s KV, context %lld, weights mapped in %.2f s", c.layers, c.n_in,
        c.n_rec, o.loops, c.n_out, o.per_loop ? "per_loop" : "last_loop", kv_dtype_name(o.kv_dtype),
        (long long)o.context, now_s() - t0);
  }

  const ModelInfo& info() const override { return info_; }
  std::unique_ptr<Session> new_session() override { return std::make_unique<DorySession>(*this); }
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
                                                         L.kind == kSliding ? c.window : 0, m.o.kv_dtype));
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
  // a ring of `window` slots still holds what position n needs when it lost at most the newest position
  bool rings = M_.c.window > 0 && std::count(M_.c.kind.begin(), M_.c.kind.end(), int(kSliding)) > 0;
  if (rings && n_ > M_.c.window && n < n_ - 1) return false;
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
