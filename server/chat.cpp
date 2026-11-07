#include "chat.h"

#include <fstream>
#include <minja/chat-template.hpp>
#include <sstream>

#include "xtg/common.h"

namespace xtg {

static std::string slurp(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f.good()) return {};
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

static std::string token_text(const json& v) {
  if (v.is_string()) return v.get<std::string>();
  if (v.is_object() && v.contains("content")) return v["content"].get<std::string>();
  return {};
}

ChatTemplate::ChatTemplate(const std::string& dir) {
  std::string source = slurp(dir + "/chat_template.jinja");
  json cfg;
  std::string tc = slurp(dir + "/tokenizer_config.json");
  if (!tc.empty()) cfg = json::parse(tc);
  if (source.empty() && cfg.contains("chat_template")) {
    const json& t = cfg["chat_template"];
    if (t.is_string()) source = t.get<std::string>();
    if (t.is_array())
      for (const auto& e : t)
        if (e.value("name", "") == "default") source = e.value("template", "");
  }
  XTG_CHECK(!source.empty(), "%s: no chat template (chat_template.jinja or tokenizer_config.json)", dir.c_str());
  if (cfg.contains("bos_token")) bos_ = token_text(cfg["bos_token"]);
  if (cfg.contains("eos_token")) eos_ = token_text(cfg["eos_token"]);
  tmpl_ = std::make_unique<minja::chat_template>(source, bos_, eos_);
}

ChatTemplate::~ChatTemplate() = default;

json ChatTemplate::prepare_messages(const json& messages) {
  json out = json::array();
  for (const auto& m : messages) {
    json msg = m;
    if (msg.contains("content") && msg["content"].is_array()) {
      std::string text;
      for (const auto& part : msg["content"])
        if (part.value("type", "") == "text") text += (text.empty() ? "" : "\n") + part.value("text", "");
      msg["content"] = text;
    }
    if (msg.value("role", "") == "assistant" && msg.contains("tool_calls") && msg["tool_calls"].is_array())
      for (auto& call : msg["tool_calls"]) {
        if (!call.contains("function")) continue;
        json& fn = call["function"];
        if (fn.contains("arguments") && fn["arguments"].is_string()) {
          json args = json::parse(fn["arguments"].get<std::string>(), nullptr, false);
          fn["arguments"] = args.is_discarded() ? json::object() : args;
        }
      }
    out.push_back(std::move(msg));
  }
  return out;
}

std::string ChatTemplate::render(const json& messages, const json& tools, bool add_generation_prompt,
                                 const json& kwargs) const {
  minja::chat_template_inputs in;
  in.messages = prepare_messages(messages);
  in.tools = tools.is_array() ? tools : json::array();
  in.add_generation_prompt = add_generation_prompt;
  in.extra_context = kwargs.is_object() ? kwargs : json::object();
  in.now = std::chrono::system_clock::from_time_t(0);
  minja::chat_template_options opts;
  opts.apply_polyfills = false;
  return tmpl_->apply(in, opts);
}

}  // namespace xtg
