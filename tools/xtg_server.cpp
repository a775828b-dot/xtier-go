// xtg-server: the server without the app (Linux development machines, adb shell on the phone).
//
//   xtg-server --model DIR [--host 127.0.0.1] [--port 18400] [--served dory] [--threads N] [--cpus 0-7]
//              [--api-key KEY] [--opt key=value ...] [--kwarg key=value ...] [--no-warmup]
//
// --opt: the pack's launch options (Dory: loops, kv_mode, kv_dtype, context, prefill_chunk, snapshots);
// --kwarg: chat_template_kwargs defaults (Dory: medium_effort=true, truncate_history_thinking=false).
#include <csignal>
#include <cstdio>
#include <string>
#include <thread>

#include "runtime.h"
#include "xtg/common.h"

using namespace xtg;

static volatile std::sig_atomic_t g_stop = 0;

static json scalar(const std::string& v) {
  if (v == "true") return true;
  if (v == "false") return false;
  return v;
}

static std::vector<int> cpu_list(const std::string& s) {
  std::vector<int> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t j = s.find(',', i);
    std::string part = s.substr(i, j == std::string::npos ? std::string::npos : j - i);
    size_t dash = part.find('-');
    int lo = std::stoi(part.substr(0, dash)), hi = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
    for (int c = lo; c <= hi; ++c) out.push_back(c);
    if (j == std::string::npos) break;
    i = j + 1;
  }
  return out;
}

int main(int argc, char** argv) {
  try {
    LaunchConfig cfg;
    for (int i = 1; i < argc; ++i) {
      std::string k = argv[i];
      if (k == "--no-warmup") {
        cfg.warmup = false;
        continue;
      }
      XTG_CHECK(i + 1 < argc, "%s needs a value", k.c_str());
      std::string v = argv[++i];
      if (k == "--model")
        cfg.model_dir = v;
      else if (k == "--host")
        cfg.host = v;
      else if (k == "--port")
        cfg.port = std::stoi(v);
      else if (k == "--served")
        cfg.served = v;
      else if (k == "--threads")
        cfg.threads = std::stoi(v);
      else if (k == "--cpus")
        cfg.cpus = cpu_list(v);
      else if (k == "--api-key")
        cfg.api_key = v;
      else if (k == "--opt" || k == "--kwarg") {
        size_t eq = v.find('=');
        XTG_CHECK(eq != std::string::npos, "%s wants key=value", k.c_str());
        (k == "--opt" ? cfg.options : cfg.template_kwargs)[v.substr(0, eq)] = scalar(v.substr(eq + 1));
      } else
        fail("unknown argument %s", k.c_str());
    }
    XTG_CHECK(!cfg.model_dir.empty(), "usage: xtg-server --model DIR [options] (see the source header)");
    std::signal(SIGINT, [](int) { g_stop = 1; });
    std::signal(SIGTERM, [](int) { g_stop = 1; });
    Runtime rt(cfg);
    rt.start();
    while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(200));
    log("stopping");
    rt.stop();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "xtg-server: %s\n", e.what());
    return 1;
  }
}
