// backend=npu: Dory on the Hexagon NPU. The model is a .pte of ExecuTorch's Qualcomm static LLM flow (built by
// tools/npu_et from the BF16 checkpoint: 16-bit activations, 4-bit weight blocks, 8-bit KV), run by libxtg_npu
// (npu/xtg_npu.h), which this file loads with dlopen: the engine builds and runs without ExecuTorch, and the app
// ships the plugin and the QNN libraries only in builds that have them. The checkpoint directory holds config.json,
// the tokenizer and the chat template as for the other backends, plus the .pte (option npu_model).
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
  a.close = reinterpret_cast<decltype(a.close)>(sym("xtg_npu_close"));
  api = a;
  return api;
}

// the .pte to run: the option, else the only one in the directory
std::string find_pte(const std::string& dir, const json& o) {
  std::string name = o.contains("npu_model") && o["npu_model"].is_string() ? o["npu_model"].get<std::string>() : "";
  if (!name.empty()) return name[0] == '/' ? name : dir + "/" + name;
  std::vector<std::string> found;
  if (DIR* d = opendir(dir.c_str())) {
    while (dirent* e = readdir(d)) {
      std::string n = e->d_name;
      if (n.size() > 4 && n.compare(n.size() - 4, 4, ".pte") == 0) found.push_back(n);
    }
    closedir(d);
  }
  XTG_CHECK(found.size() == 1, "backend=npu: %zu .pte files in %s (set npu_model)", found.size(), dir.c_str());
  return dir + "/" + found[0];
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
  NpuModel(const std::string& dir, const json& config, const json& o) {
    const std::string lib = o.contains("npu_plugin") && o["npu_plugin"].is_string()
                                ? o["npu_plugin"].get<std::string>() : "libxtg_npu.so";
    api_ = &plugin(lib);
    pte_ = find_pte(dir, o);
    char err[1024] = {0};
    double t0 = now_s();
    h_ = api_->open(pte_.c_str(), o.dump().c_str(), err, sizeof(err));
    XTG_CHECK(h_, "backend=npu: %s", err);
    char buf[2048];
    XTG_CHECK(api_->info(h_, buf, sizeof(buf)) > 0, "backend=npu: info");
    meta_ = json::parse(buf);
    chunk_ = meta_.value("prefill_ar", 128);
    info_.model_type = "dory";
    info_.vocab = meta_.at("vocab").get<int>();
    info_.context = meta_.at("context").get<int64_t>();
    info_.eos_ids = read_eos(dir, config);
    // the loops this export runs (a .pte may run fewer than config.json says): one KV instance per attention
    // evaluation, input + loops x recurrent + output
    int loops = config.value("n_recurrent_loops", 2);
    if (config.contains("layer_types")) {
      const json& t = config["layer_types"];
      const int ni = config.value("n_input_layers", 0), nr = config.value("n_recurrent_layers", 0);
      int a_in = 0, a_rec = 0, a_out = 0;
      for (int i = 0; i < int(t.size()); ++i)
        (i < ni ? a_in : i < ni + nr ? a_rec : a_out) += t[size_t(i)].get<std::string>() != "mlp";
      const int kv = meta_.value("kv_instances", 0);
      if (a_rec > 0 && kv > a_in + a_out && (kv - a_in - a_out) % a_rec == 0) loops = (kv - a_in - a_out) / a_rec;
    }
    const size_t slash = pte_.find_last_of('/');
    info_.describe = json{{"backend", "npu"},
                          {"kernels", "Hexagon HTP · 16a4w"},
                          {"pte", slash == std::string::npos ? pte_ : pte_.substr(slash + 1)},
                          {"loops", loops},
                          {"decode_rows", meta_.value("decode_ar", 1)},
                          {"kv_mode", config.value("recurrent_kv_cache_mode", std::string("per_loop"))},
                          {"kv_dtype", "uint8"},
                          {"context", info_.context},
                          {"prefill_chunk", chunk_},
                          {"kv_instances", meta_.value("kv_instances", 0)},
                          {"kv_bytes", meta_.value("kv_bytes", 0)},
                          {"threads", 1}};
    log("npu: %s loaded in %.1f s (context %lld, prefill chunk %d, %d KV instances)", pte_.c_str(), now_s() - t0,
        (long long)info_.context, chunk_, meta_.value("kv_instances", 0));
  }
  ~NpuModel() override {
    if (h_) api_->close(h_);
  }
  const ModelInfo& info() const override { return info_; }
  std::unique_ptr<Session> new_session() override { return std::make_unique<NpuSession>(*this); }

  const Api* api_ = nullptr;
  xtg_npu* h_ = nullptr;
  std::string pte_;
  json meta_;
  ModelInfo info_;
  int chunk_ = 128;
};

int64_t NpuSession::n_past() const { return M_.api_->pos(M_.h_); }

void NpuSession::forward(const int32_t* tokens, int n, Logits want, const LogitsFn& on_logits) {
  XTG_CHECK(n_past() + n <= M_.info_.context, "context full: %lld + %d > %lld tokens", (long long)n_past(), n,
            (long long)M_.info_.context);
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
  // the .pte (weights, mapped) and the plugin's caches in the layout of tools/npu_et: a full-attention instance
  // holds the context, a sliding one a ring of window + 64 slots, and each ring snapshot copies the rings. The
  // context and the loops come from the file name (tools/npu_et/export_dory.sh: ...-c4096.pte, ...-c32k.pte,
  // ...-l1-...): the estimate runs before the .pte is opened.
  const std::string path = find_pte(dir, options);
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  const size_t pte = f.good() ? size_t(f.tellg()) : 0;
  const std::string name = path.substr(path.find_last_of('/') + 1);
  int64_t ctx = 4096;
  int loops = config.value("n_recurrent_loops", 2);
  for (size_t i = 0; i + 2 < name.size(); ++i) {
    if (name[i] != '-' || !std::isdigit(static_cast<unsigned char>(name[i + 2]))) continue;
    const int64_t v = std::atoll(name.c_str() + i + 2);
    size_t j = i + 2;
    while (j < name.size() && std::isdigit(static_cast<unsigned char>(name[j]))) ++j;
    if (name[i + 1] == 'c') ctx = j < name.size() && (name[j] == 'k' || name[j] == 'K') ? v * 1024 : v;
    if (name[i + 1] == 'l' && j < name.size() && name[j] == '-') loops = int(v);
  }
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
