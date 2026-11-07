// HTTP: OpenAI-compatible chat (what DSH Go X calls) plus the part of SGLang's API that XTier's bench scripts use.
//
//   GET  /health                  200 once online (503 while loading)
//   GET  /v1/models
//   POST /v1/chat/completions     stream, tools, tool_choice, stream_options.include_usage, chat_template_kwargs
//   POST /generate                SGLang: input_ids | text, sampling_params, return_logprob, logprob_start_len,
//                                 top_logprobs_num -> meta_info.input_token_logprobs (bench/ppl_eval.py)
//   POST /flush_cache             drop the cached prefix
//   GET  /metrics                 Prometheus text
//   GET  /xtg/status              loading phase, options, telemetry (the launcher polls it)
#pragma once

#include <memory>
#include <string>
#include <thread>

namespace httplib {
class Server;
}

namespace xtg {

class Runtime;

class HttpServer {
 public:
  explicit HttpServer(Runtime& rt);
  ~HttpServer();
  void start(const std::string& host, int port);   // throws when the port cannot be bound
  void stop();

 private:
  void routes();
  Runtime& rt_;
  std::unique_ptr<httplib::Server> srv_;
  std::thread thread_;
};

}  // namespace xtg
