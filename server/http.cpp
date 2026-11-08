#include "http.h"

#include <httplib/httplib.h>

#include <chrono>
#include <ctime>
#include <random>

#include "runtime.h"
#include "xtg/common.h"

namespace xtg {

namespace {

std::string new_id(const char* prefix) {
  thread_local std::mt19937_64 rng(std::random_device{}());
  static const char* hex = "0123456789abcdef";
  std::string id = prefix;
  for (int i = 0; i < 24; ++i) id += hex[rng() & 15];
  return id;
}

// invalid UTF-8 (a scored token can be half a character) becomes U+FFFD instead of an exception
std::string dump(const json& j) { return j.dump(-1, ' ', false, json::error_handler_t::replace); }

void send_json(httplib::Response& res, const json& j, int status = 200) {
  res.status = status;
  res.set_content(dump(j), "application/json");
}

void send_error(httplib::Response& res, int status, const std::string& message, const char* type = "invalid_request_error") {
  send_json(res, json{{"error", {{"message", message}, {"type", type}, {"code", status}}}}, status);
}

std::vector<std::string> stop_list(const json& v) {
  std::vector<std::string> out;
  if (v.is_string()) out.push_back(v.get<std::string>());
  if (v.is_array())
    for (const auto& s : v)
      if (s.is_string()) out.push_back(s.get<std::string>());
  return out;
}

// sampling: request value, else the checkpoint's generation_config default, else the OpenAI default
void read_sampling(const json& src, const json& defaults, GenRequest& r) {
  auto num = [&](const char* k, double fallback) {
    if (src.contains(k) && src[k].is_number()) return src[k].get<double>();
    if (defaults.contains(k) && defaults[k].is_number()) return defaults[k].get<double>();
    return fallback;
  };
  r.sampling.temperature = float(num("temperature", 1.0));
  r.sampling.top_p = float(num("top_p", 1.0));
  r.sampling.top_k = int(num("top_k", 0));
  if (r.sampling.top_k < 0) r.sampling.top_k = 0;
  r.sampling.min_p = float(num("min_p", 0.0));
  if (src.contains("seed") && src["seed"].is_number_integer()) {
    r.has_seed = true;
    r.seed = src["seed"].get<uint64_t>();
  }
}

json chunk(const std::string& id, const std::string& model, long created, const json& delta, const json& finish) {
  return json{{"id", id},
              {"object", "chat.completion.chunk"},
              {"created", created},
              {"model", model},
              {"choices", json::array({json{{"index", 0}, {"delta", delta}, {"finish_reason", finish}}})}};
}

json delta_json(const Delta& d) {
  json j = json::object();
  if (!d.reasoning.empty()) j["reasoning_content"] = d.reasoning;
  if (!d.content.empty()) j["content"] = d.content;
  if (!d.tools.empty()) {
    json calls = json::array();
    for (const auto& t : d.tools) {
      json c{{"index", t.index}};
      if (!t.id.empty()) {
        c["id"] = t.id;
        c["type"] = "function";
        c["function"] = {{"name", t.name}, {"arguments", t.args}};
      } else {
        c["function"] = {{"arguments", t.args}};
      }
      calls.push_back(c);
    }
    j["tool_calls"] = calls;
  }
  return j;
}

}  // namespace

HttpServer::HttpServer(Runtime& rt) : rt_(rt), srv_(std::make_unique<httplib::Server>()) {}

HttpServer::~HttpServer() { stop(); }

void HttpServer::start(const std::string& host, int port) {
  routes();
  srv_->set_payload_max_length(size_t(64) << 20);
  srv_->set_read_timeout(600);
  srv_->set_write_timeout(600);
  XTG_CHECK(srv_->bind_to_port(host, port), "cannot listen on %s:%d", host.c_str(), port);
  thread_ = std::thread([this] { srv_->listen_after_bind(); });
}

void HttpServer::stop() {
  if (srv_) srv_->stop();
  if (thread_.joinable()) thread_.join();
}

void HttpServer::routes() {
  httplib::Server& s = *srv_;
  Runtime& rt = rt_;

  auto authorized = [&rt](const httplib::Request& req, httplib::Response& res) {
    const std::string& key = rt.config().api_key;
    if (key.empty() || req.get_header_value("Authorization") == "Bearer " + key) return true;
    send_error(res, 401, "invalid API key", "authentication_error");
    return false;
  };

  s.Get("/health", [&rt](const httplib::Request&, httplib::Response& res) {
    res.status = rt.online() ? 200 : 503;
    res.set_content(rt.online() ? "ok" : "loading", "text/plain");
  });

  s.Get("/xtg/status", [&rt](const httplib::Request&, httplib::Response& res) { send_json(res, rt.status()); });

  s.Get("/metrics", [&rt](const httplib::Request&, httplib::Response& res) {
    res.set_content(rt.metrics(), "text/plain; version=0.0.4");
  });

  s.Get("/v1/models", [&rt, authorized](const httplib::Request& req, httplib::Response& res) {
    if (!authorized(req, res)) return;
    json m{{"id", rt.config().served}, {"object", "model"}, {"created", long(std::time(nullptr))}, {"owned_by", "xtier-go"}};
    if (rt.online()) m["max_model_len"] = rt.info().context;
    send_json(res, json{{"object", "list"}, {"data", json::array({m})}});
  });

  s.Post("/flush_cache", [&rt, authorized](const httplib::Request& req, httplib::Response& res) {
    if (!authorized(req, res)) return;
    rt.flush_cache();
    res.set_content("Cache flushed.\n", "text/plain");
  });

  s.Post("/generate", [&rt, authorized](const httplib::Request& req, httplib::Response& res) {
    if (!authorized(req, res)) return;
    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded()) return send_error(res, 400, "body is not JSON");
    if (!rt.online()) return send_error(res, 503, "the model is not online yet", "unavailable");
    GenRequest r;
    if (body.contains("input_ids"))
      r.prompt = body["input_ids"].get<std::vector<int32_t>>();
    else if (body.contains("text") && body["text"].is_string())
      r.prompt = rt.tokenizer().encode(body["text"].get<std::string>());
    else
      return send_error(res, 400, "input_ids or text is required");
    json sp = body.value("sampling_params", json::object());
    read_sampling(sp, rt.defaults(), r);
    r.max_tokens = sp.value("max_new_tokens", 128);
    r.stop = stop_list(sp.value("stop", json()));
    r.parse = false;
    r.logprobs = body.value("return_logprob", false);
    r.logprob_start = r.logprobs ? body.value("logprob_start_len", -1) : -1;
    r.top_logprobs = body.value("top_logprobs_num", 0);
    std::string why;
    auto job = rt.submit(std::move(r), &why);
    if (!job) return send_error(res, 503, why, "unavailable");
    Event e;
    while (!(job->pop(e, 1000) && e.done)) {
    }
    if (e.finish == "error") return send_error(res, 500, e.error, "server_error");
    const json& meta = e.meta;
    json out{{"text", meta["text"]},
             {"output_ids", meta["output_ids"]},
             {"meta_info",
              {{"id", new_id("")},
               {"finish_reason", {{"type", e.finish == "length" ? "length" : "stop"}}},
               {"prompt_tokens", e.usage["prompt_tokens"]},
               {"completion_tokens", e.usage["completion_tokens"]},
               {"cached_tokens", meta["cached_tokens"]},
               {"input_token_logprobs", meta["input_token_logprobs"]},
               {"input_top_logprobs", meta["input_top_logprobs"]}}}};
    send_json(res, out);
  });

  s.Post("/v1/chat/completions", [&rt, authorized](const httplib::Request& req, httplib::Response& res) {
    if (!authorized(req, res)) return;
    json body = json::parse(req.body, nullptr, false);
    if (body.is_discarded()) return send_error(res, 400, "body is not JSON");
    if (!body.contains("messages") || !body["messages"].is_array() || body["messages"].empty())
      return send_error(res, 400, "messages is required");
    if (!rt.online()) return send_error(res, 503, "the model is not online yet", "unavailable");
    GenRequest r;
    json tools = body.contains("tools") && body["tools"].is_array() ? body["tools"] : json::array();
    std::string tool_choice = body.contains("tool_choice") && body["tool_choice"].is_string()
                                  ? body["tool_choice"].get<std::string>()
                                  : std::string("auto");
    json kwargs = rt.config().template_kwargs;
    if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
      for (auto it = body["chat_template_kwargs"].begin(); it != body["chat_template_kwargs"].end(); ++it)
        kwargs[it.key()] = it.value();
    std::string prompt;
    try {
      prompt = rt.chat().render(body["messages"], tools, true, kwargs);
    } catch (const std::exception& e) {
      return send_error(res, 400, std::string("chat template: ") + e.what());
    }
    r.prompt = rt.tokenizer().encode(prompt);
    // the template opens the reasoning ("<think>\n") unless enable_thinking is false
    const std::string open = "<think>\n";
    r.thinking = prompt.size() >= open.size() && prompt.compare(prompt.size() - open.size(), open.size(), open) == 0;
    r.tools = tool_choice == "none" ? json::array() : tools;
    read_sampling(body, rt.defaults(), r);
    int64_t room = rt.info().context - int64_t(r.prompt.size());
    int max_tokens = int(std::min<int64_t>(room, 1 << 30));
    if (body.contains("max_completion_tokens") && body["max_completion_tokens"].is_number_integer())
      max_tokens = body["max_completion_tokens"].get<int>();
    else if (body.contains("max_tokens") && body["max_tokens"].is_number_integer())
      max_tokens = body["max_tokens"].get<int>();
    r.max_tokens = max_tokens;
    r.stop = stop_list(body.value("stop", json()));
    bool stream = body.value("stream", false);
    bool usage = body.contains("stream_options") && body["stream_options"].value("include_usage", false);
    std::string why;
    auto job = rt.submit(std::move(r), &why);
    if (!job) return send_error(res, why.find("not online") != std::string::npos ? 503 : 400, why, "invalid_request_error");
    const std::string id = new_id("chatcmpl-"), model = rt.config().served;
    const long created = long(std::time(nullptr));
    if (!stream) {
      Event e;
      while (!(job->pop(e, 1000) && e.done)) {
      }
      if (e.finish == "error") return send_error(res, 500, e.error, "server_error");
      const json& meta = e.meta;
      json msg{{"role", "assistant"}};
      bool calls = !meta["tool_calls"].empty();
      std::string content = meta["content"].get<std::string>();
      msg["content"] = content.empty() && calls ? json(nullptr) : json(content);
      if (!meta["reasoning"].get<std::string>().empty()) msg["reasoning_content"] = meta["reasoning"];
      if (calls) msg["tool_calls"] = meta["tool_calls"];
      send_json(res, json{{"id", id},
                          {"object", "chat.completion"},
                          {"created", created},
                          {"model", model},
                          {"choices", json::array({json{{"index", 0}, {"message", msg}, {"finish_reason", e.finish}}})},
                          {"usage", e.usage}});
      return;
    }
    res.set_header("Cache-Control", "no-cache");
    res.set_chunked_content_provider(
        "text/event-stream",
        [job, id, model, created, usage, first = true, last = std::chrono::steady_clock::now()](
            size_t, httplib::DataSink& sink) mutable {
          auto send = [&](const json& j) {
            std::string line = "data: " + dump(j) + "\n\n";
            last = std::chrono::steady_clock::now();
            return sink.write(line.data(), line.size());
          };
          if (first) {
            first = false;
            if (!send(chunk(id, model, created, json{{"role", "assistant"}, {"content", ""}}, nullptr))) {
              job->cancelled = true;
              return false;
            }
          }
          Event e;
          if (!job->pop(e, 200)) {
            if (!sink.is_writable()) {
              job->cancelled = true;
              return false;
            }
            // a long prefill sends nothing for minutes: an empty delta every 15 s keeps clients with a stream idle
            // timeout (DSH: 5 minutes) waiting
            if (std::chrono::steady_clock::now() - last > std::chrono::seconds(15) &&
                !send(chunk(id, model, created, json::object(), nullptr))) {
              job->cancelled = true;
              return false;
            }
            return true;
          }
          if (!e.delta.empty() && !send(chunk(id, model, created, delta_json(e.delta), nullptr))) {
            job->cancelled = true;
            return false;
          }
          if (!e.done) return true;
          if (e.finish == "error")
            send(json{{"error", {{"message", e.error}, {"type", "server_error"}}}});
          else
            send(chunk(id, model, created, json::object(), e.finish));
          if (usage && e.finish != "error")
            send(json{{"id", id}, {"object", "chat.completion.chunk"}, {"created", created}, {"model", model},
                      {"choices", json::array()}, {"usage", e.usage}});
          std::string done = "data: [DONE]\n\n";
          sink.write(done.data(), done.size());
          sink.done();
          return true;
        },
        [job](bool) { job->cancelled = true; });
  });
}

}  // namespace xtg
