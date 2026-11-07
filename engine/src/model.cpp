#include "xtg/model.h"

#include <fstream>
#include <map>
#include <mutex>

#include "xtg/common.h"

namespace xtg {

static std::map<std::string, ModelPack>& packs() {
  static std::map<std::string, ModelPack> m;
  return m;
}

void register_pack(const std::string& model_type, ModelPack pack) { packs()[model_type] = std::move(pack); }

json read_config(const std::string& model_dir) {
  std::ifstream f(model_dir + "/config.json");
  XTG_CHECK(f.good(), "%s: no config.json", model_dir.c_str());
  return json::parse(f);
}

static const ModelPack& pack_for(const json& config, const std::string& model_dir) {
  std::string type = config.value("model_type", "");
  auto it = packs().find(type);
  if (it == packs().end()) {
    std::string known;
    for (auto& [k, v] : packs()) known += (known.empty() ? "" : ", ") + k;
    fail("%s: model_type \"%s\" has no pack (packs: %s)", model_dir.c_str(), type.c_str(), known.c_str());
  }
  return it->second;
}

std::unique_ptr<Model> load_model(const std::string& model_dir, const json& options, ThreadPool& pool) {
  auto config = read_config(model_dir);
  return pack_for(config, model_dir).load(model_dir, config, options, pool);
}

json estimate_memory(const std::string& model_dir, const json& options) {
  auto config = read_config(model_dir);
  return pack_for(config, model_dir).estimate(model_dir, config, options);
}

}  // namespace xtg
