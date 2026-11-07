#include "parsers.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <random>

namespace xtg {

namespace {

const std::string kFunction = "<function=", kParameter = "<parameter=", kParamEnd = "</parameter>",
                  kFunctionEnd = "</function>";

bool is_ws(char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }

size_t skip_ws(const std::string& s) {
  size_t i = 0;
  while (i < s.size() && is_ws(s[i])) ++i;
  return i;
}

std::string trim(const std::string& s) {
  size_t b = skip_ws(s), e = s.size();
  while (e > b && is_ws(s[e - 1])) --e;
  return s.substr(b, e - b);
}

std::string lower(std::string s) {
  for (char& c : s) c = char(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// s (from position i) could still turn into `tag` as more text arrives
bool maybe_prefix(const std::string& s, size_t i, const std::string& tag) {
  size_t n = std::min(s.size() - i, tag.size());
  return s.compare(i, n, tag, 0, n) == 0;
}

// the largest cut <= n that does not split a UTF-8 sequence
size_t utf8_cut(const std::string& s, size_t n) {
  while (n > 0 && n < s.size() && (static_cast<unsigned char>(s[n]) & 0xC0) == 0x80) --n;
  return n;
}

std::string random_id(const std::string& prefix) {
  thread_local std::mt19937_64 rng(std::random_device{}());
  static const char* hex = "0123456789abcdef";
  std::string id = prefix;
  for (int i = 0; i < 24; ++i) id += hex[rng() & 15];
  return id;
}

}  // namespace

std::string json_escape(const std::string& s) {
  std::string o;
  o.reserve(s.size() + 8);
  for (unsigned char c : s) {
    switch (c) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      case '\b': o += "\\b"; break;
      case '\f': o += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          o += buf;
        } else {
          o += char(c);
        }
    }
  }
  return o;
}

std::string convert_value(const std::string& raw, const std::string& type) {
  const std::string t = lower(type), v = trim(raw);
  auto str = [&] { return "\"" + json_escape(raw) + "\""; };
  if (t == "string" || t == "str" || t == "text" || t == "varchar" || t == "char" || t == "enum") return str();
  if (t == "integer" || t.rfind("int", 0) == 0 || t.rfind("uint", 0) == 0 || t == "long" || t == "short") {
    char* end = nullptr;
    long long n = std::strtoll(v.c_str(), &end, 10);
    return !v.empty() && end && *end == 0 ? std::to_string(n) : str();
  }
  if (t == "number" || t == "num" || t.rfind("float", 0) == 0 || t.rfind("double", 0) == 0) {
    json j = json::parse(v, nullptr, false);
    return !j.is_discarded() && j.is_number() ? j.dump() : str();
  }
  if (t == "boolean" || t == "bool" || t == "binary") {
    std::string l = lower(v);
    return l == "true" ? "true" : l == "false" ? "false" : str();
  }
  if (t == "null") return v == "null" ? "null" : str();
  // object, array or no type: JSON when it is JSON, else the text
  json j = json::parse(v, nullptr, false);
  return !j.is_discarded() && !v.empty() ? j.dump() : str();
}

OutputParser::OutputParser(bool thinking, const json& tools, MarkerIds ids, std::string call_prefix)
    : state_(thinking ? State::Reasoning : State::Content),
      tools_(tools),
      tools_on_(tools.is_array() && !tools.empty()),
      ids_(ids),
      prefix_(std::move(call_prefix)) {}

std::string OutputParser::param_type(const std::string& name) const {
  for (const auto& t : tools_) {
    const json& f = t.contains("function") ? t["function"] : t;
    if (f.value("name", "") != fn_) continue;
    if (!f.contains("parameters") || !f["parameters"].is_object()) return "string";
    const json& props = f["parameters"].value("properties", json::object());
    if (!props.is_object() || !props.contains(name)) return "string";   // a parameter the schema does not list
    const json& ty = props[name].value("type", json());
    if (ty.is_string()) return ty.get<std::string>();
    if (ty.is_array())
      for (const auto& x : ty)
        if (x.is_string() && x.get<std::string>() != "null") return x.get<std::string>();
    return "";
  }
  return "string";
}

void OutputParser::emit_content(const std::string& s, Delta& out, bool final) {
  // whitespace only (between or after calls) is held: it is dropped when the output ends in calls
  if (s.empty()) return;
  bool ws = std::all_of(s.begin(), s.end(), is_ws);
  if (ws && !final) {
    pending_ws_ += s;
    return;
  }
  std::string text = pending_ws_ + s;
  pending_ws_.clear();
  out.content += text;
  all_content_ += text;
}

void OutputParser::emit_args(const std::string& piece, Delta& out) {
  if (piece.empty()) return;
  call_args_ += piece;
  if (!out.tools.empty() && out.tools.back().index == calls_)
    out.tools.back().args += piece;
  else
    out.tools.push_back(ToolDelta{calls_, "", "", piece});
}

void OutputParser::parse_call(Delta& out, bool final) {
  for (;;) {
    if (stage_ == Stage::Name) {
      size_t i = skip_ws(buf_);
      if (i == buf_.size()) return;
      if (!maybe_prefix(buf_, i, kFunction)) return;   // not a call: kept whole, handed back as text at the close
      size_t gt = buf_.find('>', i + kFunction.size());
      if (gt == std::string::npos) return;
      fn_ = trim(buf_.substr(i + kFunction.size(), gt - i - kFunction.size()));
      buf_.erase(0, gt + 1);
      stage_ = Stage::Params;
      out.tools.push_back(ToolDelta{calls_, call_id_, fn_, ""});
      continue;
    }
    if (stage_ == Stage::Params) {
      size_t i = skip_ws(buf_);
      if (i == buf_.size()) {
        buf_.clear();
        return;
      }
      if (buf_.compare(i, kParameter.size(), kParameter) == 0) {
        size_t gt = buf_.find('>', i + kParameter.size());
        if (gt == std::string::npos) return;
        param_ = trim(buf_.substr(i + kParameter.size(), gt - i - kParameter.size()));
        ptype_ = param_type(param_);
        std::string t = lower(ptype_);
        value_string_ = t == "string" || t == "str" || t == "text" || t == "varchar" || t == "char" || t == "enum";
        emit_args((n_params_ == 0 ? "{\"" : ", \"") + json_escape(param_) + "\": " + (value_string_ ? "\"" : ""), out);
        ++n_params_;
        buf_.erase(0, gt + 1);
        value_started_ = false;
        value_.clear();
        stage_ = Stage::Value;
        continue;
      }
      if (buf_.compare(i, kFunctionEnd.size(), kFunctionEnd) == 0) {
        buf_.erase(0, i + kFunctionEnd.size());
        stage_ = Stage::Done;
        continue;
      }
      if (maybe_prefix(buf_, i, kParameter) || maybe_prefix(buf_, i, kFunctionEnd)) return;
      buf_.erase(0, i + 1);   // stray text between parameters
      continue;
    }
    if (stage_ == Stage::Value) {
      if (!value_started_) {
        if (buf_.empty()) return;
        if (buf_[0] == '\n') buf_.erase(0, 1);   // the newline after <parameter=...>
        value_started_ = true;
      }
      size_t end = buf_.find(kParamEnd);
      if (end != std::string::npos || final) {
        std::string v = buf_.substr(0, end == std::string::npos ? buf_.size() : end);
        if (!v.empty() && v.back() == '\n') v.pop_back();   // the newline before </parameter>
        buf_.erase(0, end == std::string::npos ? buf_.size() : end + kParamEnd.size());
        if (value_string_)
          emit_args(json_escape(v) + "\"", out);
        else
          emit_args(convert_value(value_ + v, ptype_), out);
        value_.clear();
        stage_ = Stage::Params;
        if (end == std::string::npos) return;
        continue;
      }
      // still inside the value: stream a string, keep what could be the start of "\n</parameter>"
      size_t keep = kParamEnd.size();
      if (buf_.size() <= keep) return;
      size_t cut = utf8_cut(buf_, buf_.size() - keep);
      if (cut == 0) return;
      if (value_string_)
        emit_args(json_escape(buf_.substr(0, cut)), out);
      else
        value_ += buf_.substr(0, cut);
      buf_.erase(0, cut);
      return;
    }
    // Done: the rest up to </tool_call> is whitespace
    buf_.clear();
    return;
  }
}

void OutputParser::close_call(Delta& out) {
  if (fn_.empty()) {
    // never was a call: give the text back
    emit_content("<tool_call>" + buf_, out, true);
  } else {
    if (stage_ == Stage::Value) parse_call(out, true);
    emit_args(n_params_ == 0 ? "{}" : "}", out);
    all_calls_.push_back(
        json{{"id", call_id_}, {"type", "function"}, {"function", {{"name", fn_}, {"arguments", call_args_}}}});
    ++calls_;
  }
  buf_.clear();
  fn_.clear();
  call_args_.clear();
  n_params_ = 0;
  stage_ = Stage::Name;
}

void OutputParser::text_in_call(const std::string& text, Delta& out) {
  buf_ += text;
  parse_call(out, false);
}

void OutputParser::push(int32_t id, const std::string& text, Delta& out) {
  if (id == ids_.think_close && id >= 0) {
    if (state_ == State::Reasoning) state_ = State::Content;
    return;
  }
  if (id == ids_.think_open && id >= 0) {
    if (state_ == State::Content) state_ = State::Reasoning;
    return;
  }
  if (tools_on_ && id == ids_.tool_open && id >= 0) {
    if (state_ == State::Call) close_call(out);
    state_ = State::Call;   // also from the reasoning: a call written there still counts
    call_id_ = random_id(prefix_);
    stage_ = Stage::Name;
    buf_.clear();
    return;
  }
  if (tools_on_ && id == ids_.tool_close && id >= 0) {
    if (state_ == State::Call) {
      parse_call(out, false);
      close_call(out);
      state_ = State::Content;
    }
    return;
  }
  switch (state_) {
    case State::Reasoning:
      out.reasoning += text;
      all_reasoning_ += text;
      break;
    case State::Content:
      emit_content(text, out, false);
      break;
    case State::Call:
      text_in_call(text, out);
      break;
  }
}

void OutputParser::finish(Delta& out) {
  if (state_ == State::Call) {
    parse_call(out, false);
    close_call(out);
    state_ = State::Content;
  }
  if (!pending_ws_.empty()) {
    if (calls_ == 0) {
      out.content += pending_ws_;
      all_content_ += pending_ws_;
    }
    pending_ws_.clear();
  }
}

}  // namespace xtg
