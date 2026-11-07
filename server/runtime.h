// The serving runtime: loads one model in the background (phases for the launcher), runs requests one at a time
// on its engine thread, keeps one session's KV for the next request (prefix reuse: rewind when the sliding
// windows allow, else the newest snapshot at or before the common prefix, else from zero) and serves HTTP.
#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "chat.h"
#include "parsers.h"
#include "tokenizer.h"
#include "xtg/json.h"
#include "xtg/model.h"
#include "xtg/sampler.h"
#include "xtg/threadpool.h"

namespace xtg {

struct LaunchConfig {
  std::string model_dir;
  std::string host = "127.0.0.1";
  int port = 18400;
  std::string served = "dory";
  std::string api_key;            // empty: no check (the server listens on loopback)
  int threads = 0;                // 0: every CPU the process may use
  std::vector<int> cpus;          // optional affinity, one per thread
  json options = json::object();  // the pack's launch options (loops, kv_mode, kv_dtype, context, ...)
  json template_kwargs = json::object();   // defaults under each request's chat_template_kwargs
  int max_queue = 8;
  bool warmup = true;

  static LaunchConfig from_json(const json& j);
  json to_json() const;
};

struct GenRequest {
  std::vector<int32_t> prompt;
  SamplingParams sampling;
  bool has_seed = false;
  uint64_t seed = 0;
  int max_tokens = 256;
  std::vector<std::string> stop;
  bool parse = true;          // chat: reasoning and tool calls; /generate: plain text
  bool thinking = false;      // the prompt ends inside <think>
  json tools = json::array(); // parse tool calls against these (empty: none)
  // /generate scoring
  bool logprobs = false;
  int logprob_start = -1;
  int top_logprobs = 0;
};

struct Event {
  Delta delta;
  bool done = false;
  std::string finish;         // stop | length | tool_calls | error
  std::string error;
  json usage;
  json meta;                  // /generate: output ids, logprobs; chat: the whole message
};

class Job {
 public:
  explicit Job(GenRequest r) : req(std::move(r)) {}
  GenRequest req;
  void push(Event e);
  // Wait up to timeout_ms for the next event; false on timeout.
  bool pop(Event& e, int timeout_ms);
  std::atomic<bool> cancelled{false};

 private:
  std::mutex m_;
  std::condition_variable cv_;
  std::deque<Event> events_;
};

class HttpServer;

class Runtime {
 public:
  explicit Runtime(LaunchConfig cfg);
  ~Runtime();

  void start();   // returns at once: loading and the HTTP server run in the background
  void stop();
  bool online() const { return state_.load() == State::Online && !stopping_.load(); }
  json status() const;
  std::string metrics() const;

  // used by the HTTP layer
  std::shared_ptr<Job> submit(GenRequest r, std::string* why_not);
  void flush_cache();
  const Tokenizer& tokenizer() const { return *tok_; }
  const ChatTemplate& chat() const { return *chat_; }
  const LaunchConfig& config() const { return cfg_; }
  const ModelInfo& info() const { return model_->info(); }
  MarkerIds markers() const { return markers_; }
  json defaults() const { return defaults_; }   // the checkpoint's sampling defaults (generation_config.json)

 private:
  enum class State { Idle, Loading, Online, Failed, Stopped };
  void load();
  void engine_loop();
  void run(Job& job);
  void set_phase(const char* phase, const char* label, int progress);

  LaunchConfig cfg_;
  std::atomic<State> state_{State::Idle};
  mutable std::mutex status_mutex_;
  std::string phase_, phase_label_, error_;
  int progress_ = 0;
  double started_at_ = 0, online_at_ = 0;

  std::unique_ptr<ThreadPool> pool_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<Session> session_;
  std::unique_ptr<Tokenizer> tok_;
  std::unique_ptr<ChatTemplate> chat_;
  std::unique_ptr<HttpServer> http_;
  MarkerIds markers_;
  json defaults_ = json::object();
  std::vector<int32_t> cached_;   // the tokens whose KV the session holds
  int snap_next_ = 0;
  bool flush_ = false;

  std::thread loader_, engine_;
  mutable std::mutex queue_mutex_;
  std::condition_variable queue_cv_;
  std::deque<std::shared_ptr<Job>> queue_;
  std::atomic<bool> stopping_{false};
  std::atomic<bool> running_{false};

  // telemetry
  mutable std::mutex stats_mutex_;
  double last_decode_tok_s_ = 0, last_prefill_tok_s_ = 0, last_end_ = 0;
  int64_t requests_ = 0, prompt_tokens_ = 0, gen_tokens_ = 0, cached_tokens_ = 0;
  json last_ = json::object();
};

}  // namespace xtg
