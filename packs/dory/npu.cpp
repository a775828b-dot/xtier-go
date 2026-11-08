// backend=npu: Dory on the Hexagon NPU. The model is a .pte of ExecuTorch's Qualcomm static LLM flow (built by
// tools/npu_et from the BF16 checkpoint: 16-bit activations, 4-bit weight blocks, 8-bit KV), run by libxtg_npu
// (npu/xtg_npu.h), which this file loads with dlopen: the engine builds and runs without ExecuTorch, and the app
// ships the plugin and the QNN libraries only in builds that have them. The checkpoint directory holds config.json,
// the tokenizer and the chat template as for the other backends, plus the .pte files (option npu_model).
//
// A static graph's KV length is fixed, so a context comes in buckets, one .pte each: npu_model may list several
// (smallest context first, "a-c32k.pte,a-c96k.pte"). The session starts in the first and moves to the next one that
// fits when a request needs more room (xtg_npu_grow: the caches move over, nothing is computed again); the model
// reports the last bucket's context.
#include <dirent.h>
#include <dlfcn.h>

#include <cctype>
#include <fstream>
#include <mutex>

#include "dory.h"
#include "xtg/common.h"
#include "xtg_npu.h"
#include "xtg/model.h"

namespace xtg::dory {
namespace {

struct Api {
  decltype(&xtg_npu_open) open = nullptr;
  decltype(&xtg_npu_info) info = nullptr;
  decltype(&xtg_npu_forward) forward = nullptr;
  decltype(&xtg_npu_pos) pos = nullptr;
  decltype(&xtg_npu_rewind) rewind = nullptr;
  decltype(&xtg_npu_snapshot_slots) snapshot_slots = nullptr;
  decltype(&xtg_npu_snapshot_pos) snapshot_pos = nullptr;
  decltype(&xtg_npu_snapshot) snapshot = nullptr;
  decltype(&xtg_npu_restore) restore = nullptr;
  decltype(&xtg_npu_grow) grow = nullptr;   // optional: older plugins run one bucket
  decltype(&xtg_npu_close) close = nullptr;
};

const Api& plugin(const std::string& path) {
  static std::mutex mu;
  static Api api;
  std::lock_guard<std::mutex> lock(mu);
  if (api.open) return api;
  void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  XTG_CHECK(lib, "backend=npu: cannot load %s (%s); it comes with the QNN build of the app (docs/build.md)",
            path.c_str(), dlerror());
  auto sym = [&](const char* name) {
    void* p = dlsym(lib, name);
    XTG_CHECK(p, "%s: no %s", path.c_str(), name);
    return p;
  };
  // the HTP skel runs on the DSP, which finds it through ADSP_LIBRARY_PATH: the plugin's directory (the app's
  // native libraries, where the QNN libraries are) before the system's places, unless the caller set it
  if (!getenv("ADSP_LIBRARY_PATH")) {
    Dl_info di;
    if (dladdr(sym("xtg_npu_open"), &di) && di.dli_fname) {
      std::string d = di.dli_fname;
      d = d.substr(0, d.find_last_of('/'));
      setenv("ADSP_LIBRARY_PATH", (d + ";/vendor/dsp/cdsp;/vendor/lib/rfsa/adsp;/system/lib/rfsa/adsp;/dsp").c_str(), 1);
      log("npu: ADSP_LIBRARY_PATH=%s", getenv("ADSP_LIBRARY_PATH"));
    }
  }
  Api a;
  a.open = reinterpret_cast<decltype(a.open)>(sym("xtg_npu_open"));
  a.info = reinterpret_cast<decltype(a.info)>(sym("xtg_npu_info"));
  a.forward = reinterpret_cast<decltype(a.forward)>(sym("xtg_npu_forward"));
  a.pos = reinterpret_cast<decltype(a.pos)>(sym("xtg_npu_pos"));
  a.rewind = reinterpret_cast<decltype(a.rewind)>(sym("xtg_npu_rewind"));
  a.snapshot_slots = reinterpret_cast<decltype(a.snapshot_slots)>(sym("xtg_npu_snapshot_slots"));
  a.snapshot_pos = reinterpret_cast<decltype(a.snapshot_pos)>(sym("xtg_npu_snapshot_pos"));
  a.snapshot = reinterpret_cast<decltype(a.snapshot)>(sym("xtg_npu_snapshot"));
  a.restore = reinterpret_cast<decltype(a.restore)>(sym("xtg_npu_restore"));
  a.grow = reinterpret_cast<decltype(a.grow)>(dlsym(lib, "xtg_npu_grow"));
  a.close = reinterpret_cast<decltype(a.close)>(sym("xtg_npu_close"));
  api = a;
  return api;
}

// the .pte files to run, smallest context first: the option (comma separated), else the only one in the directory
std::vector<std::string> find_ptes(const std::string& dir, const json& o) {
  std::string names = o.contains("npu_model") && o["npu_model"].is_string() ? o["npu_model"].get<std::string>() : "";
  std::vector<std::string> out;
  for (size_t a = 0; a < names.size();) {
    size_t b = names.find(',', a);
    if (b == std::string::npos) b = names.size();
    std::string n = names.substr(a, b - a);
    while (!n.empty() && n.front() == ' ') n.erase(n.begin());
    while (!n.empty() && n.back() == ' ') n.pop_back();
    if (!n.empty()) out.push_back(n[0] == '/' ? n : dir + "/" + n);
    a = b + 1;
  }
  if (!out.empty()) return out;
  std::vector<std::string> found;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      std::string n = e->d_name;
      if (n.size() > 4 && n.compare(n.size() - 4, 4, ".pte") == 0) found.push_back(n);
    }
    closedir(d);
  }
  XTG_CHECK(found.size() == 1, "backend=npu: %zu .pte files in %s (set npu_model)", found.size(), dir.c_str());
  return {dir + "/" + found[0]};
}

std::string base_name(const std::string& path) { return path.substr(path.find_last_of('/') + 1); }

// what tools/npu_et/export_dory.sh names say: ...-c4096.pte / ...-c32k.pte (the context), ...-l1-... (the loops)
void name_hints(const std::string& path, int64_t* ctx, int* loops) {
  const std::string name = base_name(path);
  for (size_t i = 0; i + 2 < name.size(); ++i) {
    if (name[i] != '-' || !std::isdigit(static_cast<unsigned char>(name[i + 2]))) continue;
    const int64_t v = std::atoll(name.c_str() + i + 2);
    size_t j = i + 2;
    while (j < name.size() && std::isdigit(static_cast<unsigned char>(name[j]))) ++j;
    if (name[i + 1] == 'c' && ctx) *ctx = j < name.size() && (name[j] == 'k' || name[j] == 'K') ? v * 1024 : v;
    if (name[i + 1] == 'l' && loops && j < name.size() && name[j] == '-') *loops = int(v);
  }
}

class NpuModel;

class NpuSession : public Session {
 public:
  explicit NpuSession(NpuModel& m) : M_(m) {}
  int64_t n_past() const override;
  void forward(const int32_t* tokens, int n, Logits want, const LogitsFn& on_logits) override;
  bool rewind(int64_t n) override;
  // full-attention caches keep every position; the sliding rings are what the snapshots keep (in the plugin)
  void snapshot(int slot) override;
  bool restore(int slot) override;
  int64_t snapshot_pos(int slot) const override;
  int snapshot_slots() const override;
  void reset() override { rewind(0); }
  size_t kv_bytes() const override;

 private:
  NpuModel& M_;
};

class NpuModel : public Model {
 public:
  NpuModel(const std::string& dir, const json& config, const json& o) : config_(config) {
    const std::string lib = o.contains("npu_plugin") && o["npu_plugin"].is_string()
                                ? o["npu_plugin"].get<std::string>() : "libxtg_npu.so";
    api_ = &plugin(lib);
    for (const std::string& p : find_ptes(dir, o)) {
      int64_t ctx = 0;
      name_hints(p, &ctx, nullptr);
      buckets_.push_back({p, ctx});
    }
    XTG_CHECK(buckets_.size() == 1 || api_->grow, "backend=npu: %zu buckets, the plugin cannot grow a session",
              buckets_.size());
    char err[1024] = {0};
    double t0 = now_s();
    h_ = api_->open(buckets_[0].path.c_str(), o.dump().c_str(), err, sizeof(err));
    XTG_CHECK(h_, "backend=npu: %s", err);
    refresh();
    buckets_[0].context = meta_.at("context").get<int64_t>();
    for (size_t i = 1; i < buckets_.size(); ++i)
      XTG_CHECK(buckets_[i].context > buckets_[i - 1].context, "backend=npu: %s after %s: list the buckets by "
                "context (the names say it: ...-c32k.pte)", base_name(buckets_[i].path).c_str(),
                base_name(buckets_[i - 1].path).c_str());
    chunk_ = meta_.value("prefill_ar", 128);
    info_.model_type = "dory";
    info_.vocab = meta_.at("vocab").get<int>();
    info_.context = buckets_.back().context;
    info_.eos_ids = read_eos(dir, config);
    describe();
    log("npu: %s loaded in %.1f s (context %lld of %lld, prefill chunk %d, %d KV instances, %s)",
        buckets_[0].path.c_str(), now_s() - t0, (long long)meta_.at("context").get<int64_t>(),
        (long long)info_.context, chunk_, meta_.value("kv_instances", 0),
        meta_.value("kvmap", false) ? "KV byte maps" : "no KV byte maps: NAME.kvmap.json is missing, cached V may be off");
  }
  ~NpuModel() override {
    if (h_) api_->close(h_);
  }
  const ModelInfo& info() const override { return info_; }
  std::unique_ptr<Session> new_session() override { return std::make_unique<NpuSession>(*this); }

  // the session in the smallest bucket with room for `need` positions: up when a request outgrows the current one,
  // down when a shorter conversation follows (a static graph pays for its whole context every step)
  void fit(int64_t need) {
    size_t to = 0;
    while (to < buckets_.size() && buckets_[to].context < need) ++to;
    XTG_CHECK(to < buckets_.size(), "context full: %lld positions (the largest bucket holds %lld)", (long long)need,
              (long long)info_.context);
    if (to == cur_ || api_->pos(h_) > buckets_[to].context) return;
    char err[1024] = {0};
    const double t0 = now_s();
    const int rc = api_->grow(h_, buckets_[to].path.c_str(), err, sizeof(err));
    XTG_CHECK(rc == 0, "backend=npu: moving to %s: %s", base_name(buckets_[to].path).c_str(), err);
    cur_ = to;
    refresh();
    describe();
    log("npu: %lld positions moved to %s (context %lld) in %.1f s", (long long)api_->pos(h_),
        base_name(buckets_[to].path).c_str(), (long long)meta_.at("context").get<int64_t>(), now_s() - t0);
  }

  const Api* api_ = nullptr;
  xtg_npu* h_ = nullptr;
  json config_, meta_;
  ModelInfo info_;
  int chunk_ = 128;
  struct Bucket {
    std::string path;
    int64_t context = 0;
  };
  std::vector<Bucket> buckets_;
  size_t cur_ = 0;

 private:
  void refresh() {
    char buf[2048];
    XTG_CHECK(api_->info(h_, buf, sizeof(buf)) > 0, "backend=npu: info");
    meta_ = json::parse(buf);
  }

  void describe() {
    // the loops this export runs (a .pte may run fewer than config.json says): one KV instance per attention
    // evaluation, input + loops x recurrent + output
    int loops = config_.value("n_recurrent_loops", 2);
    if (config_.contains("layer_types")) {
      const json& t = config_["layer_types"];
      const int ni = config_.value("n_input_layers", 0), nr = config_.value("n_recurrent_layers", 0);
      int a_in = 0, a_rec = 0, a_out = 0;
      for (int i = 0; i < int(t.size()); ++i)
        (i < ni ? a_in : i < ni + nr ? a_rec : a_out) += t[size_t(i)].get<std::string>() != "mlp";
      const int kv = meta_.value("kv_instances", 0);
      if (a_rec > 0 && kv > a_in + a_out && (kv - a_in - a_out) % a_rec == 0) loops = (kv - a_in - a_out) / a_rec;
    }
    json names = json::array();
    for (const Bucket& b : buckets_) names.push_back(base_name(b.path));
    info_.describe = json{{"backend", "npu"},
                          {"kernels", "Hexagon HTP · 16a4w"},
                          {"pte", base_name(buckets_[cur_].path)},
                          {"buckets", names},
                          {"bucket_context", meta_.at("context").get<int64_t>()},
                          {"loops", loops},
                          {"decode_rows", meta_.value("decode_ar", 1)},
                          {"kv_mode", config_.value("recurrent_kv_cache_mode", std::string("per_loop"))},
                          {"kv_dtype", "uint8"},
                          {"context", info_.context},
                          {"prefill_chunk", chunk_},
                          {"kv_instances", meta_.value("kv_instances", 0)},
                          {"kv_bytes", meta_.value("kv_bytes", 0)},
                          {"kvmap", meta_.value("kvmap", false)},
                          {"threads", 1}};
  }
};

int64_t NpuSession::n_past() const { return M_.api_->pos(M_.h_); }

void NpuSession::forward(const int32_t* tokens, int n, Logits want, const LogitsFn& on_logits) {
  const int64_t need = n_past() + n;
  XTG_CHECK(need <= M_.info_.context, "context full: %lld + %d > %lld tokens", (long long)n_past(), n,
            (long long)M_.info_.context);
  if (M_.buckets_.size() > 1) M_.fit(need);
  struct Ctx {
    const LogitsFn* fn;
    int off;
  };
  // the plugin runs a chunk of up to one prefill AR (short runs token by token); progress and cancellation per chunk
  for (int off = 0; off < n; off += M_.chunk_) {
    check_cancelled();
    report_progress(off);
    int m = std::min(M_.chunk_, n - off);
    int w = want == Logits::All ? 2 : (want == Logits::Last && off + m == n ? 1 : 0);
    Ctx c{&on_logits, off};
    char err[1024] = {0};
    int rc = M_.api_->forward(M_.h_, tokens + off, m, w,
                              [](int row, const float* lg, void* u) {
                                auto* c = static_cast<Ctx*>(u);
                                (*c->fn)(c->off + row, lg);
                              },
                              &c, err, sizeof(err));
    XTG_CHECK(rc == 0, "backend=npu: %s", err);
  }
}

bool NpuSession::rewind(int64_t n) { return M_.api_->rewind(M_.h_, n) == 0; }

void NpuSession::snapshot(int slot) { M_.api_->snapshot(M_.h_, slot); }

bool NpuSession::restore(int slot) { return M_.api_->restore(M_.h_, slot) == 0; }

int64_t NpuSession::snapshot_pos(int slot) const { return M_.api_->snapshot_pos(M_.h_, slot); }

int NpuSession::snapshot_slots() const { return M_.api_->snapshot_slots(M_.h_); }

size_t NpuSession::kv_bytes() const { return M_.meta_.value("kv_bytes", size_t(0)); }

}  // namespace

std::unique_ptr<Model> load_npu(const std::string& dir, const json& config, const json& options) {
  return std::make_unique<NpuModel>(dir, config, options);
}

json estimate_npu(const std::string& dir, const json& config, const json& options) {
  // the largest bucket (the session ends up there): its .pte (weights, mapped) and the plugin's caches in the layout
  // of tools/npu_et (a full-attention instance holds the context, a sliding one a ring of window + 64 slots, each
  // ring snapshot copies the rings). The context and the loops come from the file name: the estimate runs before
  // the .pte is opened.
  const std::string path = find_ptes(dir, options).back();
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  const size_t pte = f.good() ? size_t(f.tellg()) : 0;
  int64_t ctx = 4096;
  int loops = config.value("n_recurrent_loops", 2);
  name_hints(path, &ctx, &loops);
  const int n_in = config.value("n_input_layers", 0), n_rec = config.value("n_recurrent_layers", 0);
  const int64_t ring = config.value("sliding_window", 1024) + 64;
  int full = 0, sliding = 0;
  const auto& kinds = config.at("layer_types");
  for (int l = 0; l < int(kinds.size()); ++l) {
    const std::string k = kinds[size_t(l)].get<std::string>();
    if (k == "mlp") continue;
    const int n = l >= n_in && l < n_in + n_rec ? loops : 1;
    (k == "sliding_attention" ? sliding : full) += n;
  }
  const size_t row = 2 * size_t(config.value("num_key_value_heads", 4)) * size_t(config.value("head_dim", 256));
  const size_t kv_full = size_t(full) * row * size_t(ctx), kv_ring = size_t(sliding) * row * size_t(ring);
  const size_t snaps = size_t(options.value("snapshots", 2)) * kv_ring;
  return json{{"weights_bytes", pte},  {"embedding_bytes", 0}, {"kv_full_bytes", kv_full}, {"kv_window_bytes", kv_ring},
              {"snapshot_bytes", snaps}, {"work_bytes", 0},    {"resident_bytes", pte + kv_full + kv_ring + snaps},
              {"context", ctx},        {"loops", loops}};
}

}  // namespace xtg::dory
