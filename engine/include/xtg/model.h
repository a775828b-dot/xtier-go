// What the server needs from a model pack: load a checkpoint with launch options, then run sessions (one KV state
// each). Packs are registered under the checkpoint's model_type (packs/<name>/, register_builtin_packs).
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "xtg/json.h"

#include "xtg/threadpool.h"

namespace xtg {

struct ModelInfo {
  std::string model_type;
  int vocab = 0;
  int64_t context = 0;            // the KV capacity sessions are built for
  std::vector<int> eos_ids;
  json describe;        // what /xtg/status and the startup log show
};

// Logits of the rows a forward computes: none (prefill only), the last row, or every row (scoring), handed to the
// callback row by row so a long prompt never holds [rows, vocab] at once.
enum class Logits { None, Last, All };
using LogitsFn = std::function<void(int row, const float* logits)>;

class Session {
 public:
  virtual ~Session() = default;
  void set_cancel_check(std::function<bool()> check) { cancel_check_ = std::move(check); }
  virtual int64_t n_past() const = 0;
  // Run n tokens at positions n_past() .. n_past() + n - 1 (any length; the pack cuts it into prefill chunks).
  virtual void forward(const int32_t* tokens, int n, Logits want, const LogitsFn& on_logits) = 0;
  // Go back to n positions without recomputing; false when the cache cannot (a sliding window already overwrote
  // what position n would need).
  virtual bool rewind(int64_t n) = 0;
  // Remember the sliding-window state at n_past() in slot `slot`, and go back to it later (n_past() is then the
  // snapshot's position). Writing below a snapshot's position drops the snapshot.
  virtual void snapshot(int slot) = 0;
  virtual bool restore(int slot) = 0;
  virtual int64_t snapshot_pos(int slot) const = 0;   // -1: empty
  virtual int snapshot_slots() const = 0;
  virtual void reset() = 0;
  virtual size_t kv_bytes() const = 0;                // resident estimate

 protected:
  void check_cancelled() const {
    if (cancel_check_ && cancel_check_()) throw std::runtime_error("request cancelled");
  }

 private:
  std::function<bool()> cancel_check_;
};

class Model {
 public:
  virtual ~Model() = default;
  virtual const ModelInfo& info() const = 0;
  virtual std::unique_ptr<Session> new_session() = 0;
};

struct ModelPack {
  // model_dir holds config.json and the weights; options are the launch options (the pack's card lists them).
  std::function<std::unique_ptr<Model>(const std::string& model_dir, const json& config,
                                       const json& options, ThreadPool& pool)>
      load;
  // Bytes the options would take (weights, KV at the full context, snapshots, work buffers) without loading,
  // so the launcher can grey out what does not fit.
  std::function<json(const std::string& model_dir, const json& config,
                               const json& options)>
      estimate;
};

void register_pack(const std::string& model_type, ModelPack pack);
void register_builtin_packs();
json read_config(const std::string& model_dir);
std::unique_ptr<Model> load_model(const std::string& model_dir, const json& options, ThreadPool& pool);
json estimate_memory(const std::string& model_dir, const json& options);

}  // namespace xtg
