#include <cstring>
#include <memory>
#include <mutex>

#include "runtime.h"
#include "xtg/common.h"
#include "xtg_capi.h"

namespace {
std::mutex g_mutex;
std::unique_ptr<xtg::Runtime> g_runtime;

int copy_out(const std::string& s, char* buf, int len) {
  if (buf && len > 0) {
    size_t n = std::min(s.size(), size_t(len - 1));
    std::memcpy(buf, s.data(), n);
    buf[n] = 0;
  }
  return int(s.size());
}
}  // namespace

extern "C" int xtg_start(const char* config_json, char* err, int err_len) {
  std::lock_guard<std::mutex> lock(g_mutex);
  try {
    XTG_CHECK(!g_runtime, "a server is already running in this process; stop it first");
    auto cfg = xtg::LaunchConfig::from_json(xtg::json::parse(config_json));
    auto rt = std::make_unique<xtg::Runtime>(cfg);
    rt->start();
    g_runtime = std::move(rt);
    return 0;
  } catch (const std::exception& e) {
    copy_out(e.what(), err, err_len);
    return -1;
  }
}

extern "C" int xtg_stop(void) {
  std::unique_ptr<xtg::Runtime> rt;
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    rt = std::move(g_runtime);
  }
  if (rt) rt->stop();
  return 0;
}

extern "C" int xtg_status(char* buf, int len) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_runtime) return copy_out(R"({"state":"stopped"})", buf, len);
  return copy_out(g_runtime->status().dump(), buf, len);
}

extern "C" int xtg_estimate(const char* model_dir, const char* options_json, char* buf, int len) {
  try {
    xtg::register_builtin_packs();
    auto opts = options_json && *options_json ? xtg::json::parse(options_json) : xtg::json::object();
    return copy_out(xtg::estimate_memory(model_dir, opts).dump(), buf, len);
  } catch (const std::exception& e) {
    return copy_out(xtg::json{{"error", e.what()}}.dump(), buf, len);
  }
}

extern "C" const char* xtg_version(void) { return xtg::kVersion; }
