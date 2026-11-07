#include "runtime.h"

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <fstream>
#include <random>
#include <sstream>
#include <thread>

#include "http.h"
#include "xtg/common.h"
#include "xtg/kernels.h"

namespace xtg {

// --- launch configuration ---

LaunchConfig LaunchConfig::from_json(const json& j) {
  LaunchConfig c;
  c.model_dir = j.at("model_dir").get<std::string>();
  c.host = j.value("host", c.host);
  c.port = j.value("port", c.port);
  c.served = j.value("served", c.served);
  c.api_key = j.value("api_key", std::string());
  c.threads = j.value("threads", 0);
  if (j.contains("cpus")) c.cpus = j["cpus"].get<std::vector<int>>();
  c.options = j.value("options", json::object());
  c.template_kwargs = j.value("template_kwargs", json::object());
  c.max_queue = j.value("max_queue", c.max_queue);
  c.warmup = j.value("warmup", true);
  return c;
}

json LaunchConfig::to_json() const {
  return json{{"model_dir", model_dir}, {"host", host},       {"port", port},
              {"served", served},       {"threads", threads}, {"options", options},
              {"template_kwargs", template_kwargs},          {"api_key_set", !api_key.empty()}};
}

// --- jobs ---

void Job::push(Event e) {
  {
    std::lock_guard<std::mutex> lock(m_);
    events_.push_back(std::move(e));
  }
  cv_.notify_all();
}

bool Job::pop(Event& e, int timeout_ms) {
  std::unique_lock<std::mutex> lock(m_);
  if (!cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [&] { return !events_.empty(); })) return false;
  e = std::move(events_.front());
  events_.pop_front();
  return true;
}

// Bytes of tokens -> complete UTF-8 text (a character can span tokens).
class Utf8Stream {
 public:
  std::string push(const std::string& bytes) {
    pending_ += bytes;
    size_t n = pending_.size(), cut = n;
    // the start of the last character, and whether it is complete
    size_t i = n;
    while (i > 0 && n - i < 4) {
      unsigned char c = static_cast<unsigned char>(pending_[i - 1]);
      if ((c & 0xC0) != 0x80) {
        size_t need = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (n - (i - 1) < need) cut = i - 1;
        break;
      }
      --i;
    }
    std::string out = pending_.substr(0, cut);
    pending_.erase(0, cut);
    return out;
  }
  std::string flush() {
    std::string out;
    if (!pending_.empty()) out = "\xEF\xBF\xBD";   // an incomplete character at the end: U+FFFD
    pending_.clear();
    return out;
  }

 private:
  std::string pending_;
};

// --- runtime ---

Runtime::Runtime(LaunchConfig cfg) : cfg_(std::move(cfg)) {}

Runtime::~Runtime() { stop(); }

void Runtime::set_phase(const char* phase, const char* label, int progress) {
  std::lock_guard<std::mutex> lock(status_mutex_);
  phase_ = phase;
  phase_label_ = label;
  progress_ = progress;
  log("phase %s (%d%%)", phase, progress);
}

void Runtime::start() {
  State expected = State::Idle;
  XTG_CHECK(state_.compare_exchange_strong(expected, State::Loading), "the runtime was started before");
  started_at_ = now_s();
  register_builtin_packs();
  http_ = std::make_unique<HttpServer>(*this);
  http_->start(cfg_.host, cfg_.port);   // /health and /xtg/status answer while loading
  loader_ = std::thread([this] { load(); });
}

void Runtime::load() {
  try {
    set_phase("files", "检查文件", 5);
    for (const char* f : {"/config.json", "/tokenizer.json", "/model.safetensors"}) {
      std::ifstream probe(cfg_.model_dir + f);
      XTG_CHECK(probe.good(), "%s%s is missing", cfg_.model_dir.c_str(), f);
    }
    int threads = cfg_.threads > 0 ? cfg_.threads : int(std::max(1u, std::thread::hardware_concurrency()));
    pool_ = std::make_unique<ThreadPool>(threads, cfg_.cpus);
    set_phase("tokenizer", "分词器与模板", 10);
    tok_ = std::make_unique<Tokenizer>(cfg_.model_dir + "/tokenizer.json");
    chat_ = std::make_unique<ChatTemplate>(cfg_.model_dir);
    markers_.think_open = tok_->find("<think>");
    markers_.think_close = tok_->find("</think>");
    markers_.tool_open = tok_->find("<tool_call>");
    markers_.tool_close = tok_->find("</tool_call>");
    std::ifstream g(cfg_.model_dir + "/generation_config.json");
    if (g.good()) {
      json gc = json::parse(g);
      for (const char* k : {"temperature", "top_p", "top_k", "min_p"})
        if (gc.contains(k)) defaults_[k] = gc[k];
    }
    set_phase("weights", "映射权重", 25);
    model_ = load_model(cfg_.model_dir, cfg_.options, *pool_);
    set_phase("kv", "分配 KV 缓存", 60);
    session_ = model_->new_session();
    if (cfg_.warmup) {
      // reads every weight page once and sizes the work buffers before the first client request
      set_phase("warmup", "预热", 75);
      std::vector<int32_t> ids = tok_->encode("Warm-up: the quick brown fox jumps over the lazy dog. 预热。");
      while (ids.size() < 64) ids.push_back(ids[ids.size() % 8]);
      int next = 0;
      session_->forward(ids.data(), int(ids.size()), Logits::Last, [&](int, const float* lg) { next = argmax(lg, info().vocab); });
      for (int i = 0; i < 2; ++i) {
        int32_t t = next;
        session_->forward(&t, 1, Logits::Last, [&](int, const float* lg) { next = argmax(lg, info().vocab); });
      }
      session_->reset();
    }
    engine_ = std::thread([this] { engine_loop(); });
    online_at_ = now_s();
    set_phase("online", "在线", 100);
    state_.store(State::Online);
    log("online on %s:%d as %s after %.1f s (%s kernels, %d threads)", cfg_.host.c_str(), cfg_.port, cfg_.served.c_str(),
        online_at_ - started_at_, kernel_set_name(), pool_->size());
  } catch (const std::exception& e) {
    {
      std::lock_guard<std::mutex> lock(status_mutex_);
      error_ = e.what();
    }
    state_.store(State::Failed);
    log("loading failed: %s", e.what());
  }
}

void Runtime::stop() {
  if (stopping_.exchange(true)) return;
  queue_cv_.notify_all();
  if (loader_.joinable()) loader_.join();
  if (engine_.joinable()) engine_.join();
  if (http_) http_->stop();
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    for (auto& j : queue_) {
      Event e;
      e.done = true;
      e.finish = "error";
      e.error = "server stopping";
      j->push(e);
    }
    queue_.clear();
  }
  session_.reset();
  model_.reset();
  pool_.reset();
  if (state_.load() != State::Failed) state_.store(State::Stopped);
}

std::shared_ptr<Job> Runtime::submit(GenRequest r, std::string* why_not) {
  if (!online()) {
    *why_not = "the model is not online yet";
    return nullptr;
  }
  if (r.prompt.empty()) {
    *why_not = "empty prompt";
    return nullptr;
  }
  int64_t room = info().context - int64_t(r.prompt.size());
  if (room < 1) {
    *why_not = format("the prompt has %zu tokens, the context is %lld", r.prompt.size(), (long long)info().context);
    return nullptr;
  }
  r.max_tokens = int(std::min<int64_t>(std::max(0, r.max_tokens), room));
  auto job = std::make_shared<Job>(std::move(r));
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    if (int(queue_.size()) >= cfg_.max_queue) {
      *why_not = "too many queued requests";
      return nullptr;
    }
    queue_.push_back(job);
  }
  queue_cv_.notify_one();
  return job;
}

void Runtime::flush_cache() {
  std::lock_guard<std::mutex> lock(queue_mutex_);
  flush_ = true;   // the engine thread drops the session's KV before its next request
}

void Runtime::engine_loop() {
  for (;;) {
    std::shared_ptr<Job> job;
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      queue_cv_.wait(lock, [&] { return stopping_.load() || !queue_.empty(); });
      if (stopping_.load()) return;
      job = queue_.front();
      queue_.pop_front();
      if (flush_) {
        session_->reset();
        cached_.clear();
        flush_ = false;
      }
    }
    running_.store(true);
    try {
      run(*job);
    } catch (const std::exception& e) {
      // the session may be half-written: start the next request from zero
      session_->reset();
      cached_.clear();
      Event ev;
      ev.done = true;
      ev.finish = "error";
      ev.error = e.what();
      job->push(ev);
      log("request failed: %s", e.what());
    }
    running_.store(false);
  }
}

void Runtime::run(Job& job) {
  GenRequest& r = job.req;
  const int vocab = info().vocab;
  const int64_t m = int64_t(r.prompt.size());
  // --- reuse the cached prefix ---
  int64_t reuse = 0;
  if (r.logprobs) {
    session_->reset();
    cached_.clear();
  } else {
    size_t l = 0;
    while (l < cached_.size() && l < r.prompt.size() && cached_[l] == r.prompt[l]) ++l;
    l = std::min<size_t>(l, size_t(m - 1));   // the last prompt token runs again for its logits
    if (session_->n_past() == int64_t(cached_.size()) && session_->rewind(int64_t(l))) {
      reuse = int64_t(l);
    } else {
      int best = -1;
      int64_t pos = -1;
      for (int s = 0; s < session_->snapshot_slots(); ++s) {
        int64_t p = session_->snapshot_pos(s);
        if (p >= 0 && p <= int64_t(l) && p > pos) {
          best = s;
          pos = p;
        }
      }
      if (best >= 0 && session_->restore(best)) {
        reuse = pos;
      } else {
        session_->reset();
        reuse = 0;
      }
    }
    cached_.resize(size_t(reuse));
  }
  // --- prefill ---
  Sampler sampler(r.has_seed ? r.seed : std::random_device{}());
  int next = -1;
  json in_lp = json::array(), in_top = json::array();
  if (r.logprobs && r.logprob_start >= 0) {
    if (r.logprob_start == 0) {
      in_lp.push_back(json::array({nullptr, r.prompt[0], nullptr}));
      in_top.push_back(nullptr);
    }
  }
  double t0 = now_s();
  const int64_t todo = m - reuse;
  std::vector<std::pair<float, int>> top;
  session_->forward(r.prompt.data() + reuse, int(todo), r.logprobs ? Logits::All : Logits::Last,
                    [&](int row, const float* lg) {
                      int64_t p = reuse + row;   // these logits predict position p + 1
                      if (r.logprobs && r.logprob_start >= 0 && p + 1 < m && p + 1 >= r.logprob_start) {
                        in_lp.push_back(json::array({logprob(lg, vocab, r.prompt[size_t(p + 1)]), r.prompt[size_t(p + 1)], nullptr}));
                        if (r.top_logprobs > 0) {
                          top_logprobs(lg, vocab, r.top_logprobs, top);
                          json t = json::array();
                          for (auto& [lp, id] : top) t.push_back(json::array({lp, id, nullptr}));
                          in_top.push_back(t);
                        } else {
                          in_top.push_back(nullptr);
                        }
                      }
                      if (p + 1 == m) next = sampler.sample(lg, vocab, r.sampling);
                    });
  cached_ = r.prompt;
  if (session_->snapshot_slots() > 0 && !r.logprobs) {
    session_->snapshot(snap_next_);
    snap_next_ = (snap_next_ + 1) % session_->snapshot_slots();
  }
  double t1 = now_s();
  // --- decode ---
  const auto& eos = info().eos_ids;
  OutputParser parser(r.thinking, r.parse ? r.tools : json::array(), markers_);
  Utf8Stream utf8;
  std::vector<int32_t> out_ids;
  std::string text_all;
  std::string finish = "length";
  for (int i = 0; i < r.max_tokens; ++i) {
    int32_t tok = next;
    if (std::find(eos.begin(), eos.end(), tok) != eos.end()) {
      finish = "stop";
      break;
    }
    out_ids.push_back(tok);
    Event ev;
    if (r.parse) {
      bool marker = tok == markers_.think_open || tok == markers_.think_close || tok == markers_.tool_open ||
                    tok == markers_.tool_close;
      std::string text = utf8.push(marker ? std::string() : tok_->token_bytes(tok));
      if (marker) {
        std::string tail = utf8.flush();
        if (!tail.empty()) parser.push(-1, tail, ev.delta);
      }
      parser.push(tok, text, ev.delta);
      text_all += text;
    } else {
      text_all += tok_->token_bytes(tok);
    }
    bool stopped = false;
    for (const auto& s : r.stop)
      if (!s.empty() && text_all.size() >= s.size() &&
          text_all.find(s, text_all.size() > s.size() + 64 ? text_all.size() - s.size() - 64 : 0) != std::string::npos)
        stopped = true;
    if (!ev.delta.empty()) job.push(std::move(ev));
    if (stopped) {
      finish = "stop";
      break;
    }
    if (job.cancelled.load()) {
      finish = "cancelled";
      break;
    }
    if (i + 1 == r.max_tokens) break;
    session_->forward(&tok, 1, Logits::Last, [&](int, const float* lg) { next = sampler.sample(lg, vocab, r.sampling); });
    cached_.push_back(tok);
  }
  double t2 = now_s();
  Event done;
  if (r.parse) {
    std::string tail = utf8.flush();
    if (!tail.empty()) parser.push(-1, tail, done.delta);
    parser.finish(done.delta);
    if (finish == "stop" && parser.tool_calls() > 0) finish = "tool_calls";
  }
  done.done = true;
  done.finish = finish;
  done.usage = json{{"prompt_tokens", m},
                    {"completion_tokens", out_ids.size()},
                    {"total_tokens", m + int64_t(out_ids.size())},
                    {"prompt_tokens_details", {{"cached_tokens", reuse}}}};
  done.meta = json{{"output_ids", out_ids},
                   {"reasoning", parser.reasoning()},
                   {"content", parser.content()},
                   {"tool_calls", parser.message_tool_calls()},
                   {"text", r.parse ? std::string() : text_all},
                   {"cached_tokens", reuse},
                   {"input_token_logprobs", in_lp},
                   {"input_top_logprobs", in_top}};
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    ++requests_;
    prompt_tokens_ += m;
    gen_tokens_ += int64_t(out_ids.size());
    cached_tokens_ += reuse;
    if (todo > 0 && t1 > t0) last_prefill_tok_s_ = double(todo) / (t1 - t0);
    if (out_ids.size() > 1 && t2 > t1) last_decode_tok_s_ = double(out_ids.size() - 1) / (t2 - t1);
    last_end_ = t2;
    last_ = json{{"prompt_tokens", m},        {"cached_tokens", reuse},           {"completion_tokens", out_ids.size()},
                 {"prefill_s", t1 - t0},      {"decode_s", t2 - t1},             {"prefill_tok_s", last_prefill_tok_s_},
                 {"decode_tok_s", last_decode_tok_s_}, {"finish", finish}};
  }
  log("request: prompt %lld (cached %lld), %zu generated, prefill %.1f tok/s, decode %.1f tok/s, %s", (long long)m,
      (long long)reuse, out_ids.size(), todo > 0 ? double(todo) / std::max(1e-9, t1 - t0) : 0.0,
      out_ids.size() > 1 ? double(out_ids.size() - 1) / std::max(1e-9, t2 - t1) : 0.0, finish.c_str());
  job.push(std::move(done));
}

static size_t rss_bytes() {
  std::ifstream f("/proc/self/statm");
  size_t pages = 0, resident = 0;
  f >> pages >> resident;
  return resident * size_t(::sysconf(_SC_PAGESIZE));
}

json Runtime::status() const {
  static const char* names[] = {"idle", "loading", "online", "failed", "stopped"};
  json s;
  {
    std::lock_guard<std::mutex> lock(status_mutex_);
    s["state"] = names[int(state_.load())];
    s["phase"] = phase_;
    s["phase_label"] = phase_label_;
    s["progress"] = progress_;
    s["error"] = error_;
  }
  s["version"] = kVersion;
  s["served"] = cfg_.served;
  s["port"] = cfg_.port;
  s["config"] = cfg_.to_json();
  s["uptime_s"] = started_at_ > 0 ? now_s() - started_at_ : 0.0;
  if (state_.load() == State::Online && model_) s["model"] = model_->info().describe;
  json t;
  {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    // a decode rate older than a few seconds is shown as idle
    bool idle = !running_.load() && (last_end_ == 0 || now_s() - last_end_ > 5);
    t["decode_tok_s"] = idle ? 0.0 : last_decode_tok_s_;
    t["last_decode_tok_s"] = last_decode_tok_s_;
    t["last_prefill_tok_s"] = last_prefill_tok_s_;
    t["requests"] = requests_;
    t["prompt_tokens_total"] = prompt_tokens_;
    t["generation_tokens_total"] = gen_tokens_;
    t["cached_tokens_total"] = cached_tokens_;
    t["last"] = last_;
  }
  t["running"] = running_.load();
  {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    t["queued"] = queue_.size();
  }
  t["rss_bytes"] = rss_bytes();
  s["telemetry"] = t;
  return s;
}

std::string Runtime::metrics() const {
  std::ostringstream o;
  std::lock_guard<std::mutex> lock(stats_mutex_);
  auto line = [&](const char* name, const char* type, double v) {
    o << "# TYPE xtg:" << name << " " << type << "\nxtg:" << name << " " << v << "\n";
  };
  line("requests_total", "counter", double(requests_));
  line("prompt_tokens_total", "counter", double(prompt_tokens_));
  line("generation_tokens_total", "counter", double(gen_tokens_));
  line("cached_tokens_total", "counter", double(cached_tokens_));
  line("num_running_reqs", "gauge", running_.load() ? 1 : 0);
  line("gen_throughput", "gauge", last_decode_tok_s_);
  line("prefill_throughput", "gauge", last_prefill_tok_s_);
  line("rss_bytes", "gauge", double(rss_bytes()));
  return o.str();
}

}  // namespace xtg
