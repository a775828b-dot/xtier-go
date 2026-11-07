// Generated tokens -> OpenAI deltas: reasoning (<think> ... </think>), content, and tool calls in the qwen3_coder
// XML format the Dory template asks for:
//
//   <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n...</function>\n</tool_call>
//
// <think>, </think>, <tool_call>, </tool_call> are single special tokens, so the parser switches state on token
// ids and only scans text inside a call. Arguments stream as JSON while they are generated: a string parameter's
// value goes out in pieces (a long file body is not held back until the call closes), other types when the
// parameter ends, converted by the tool's JSON schema (integer, number, boolean, object, array; else a string).
// What the 2026-11-01 review of another stack's parser found is covered: a call inside the reasoning still counts,
// a call left open at the end of the output is closed and kept, long values stream.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "xtg/json.h"

namespace xtg {

struct ToolDelta {
  int index = 0;
  std::string id;      // set on the first delta of a call
  std::string name;    // set on the first delta of a call
  std::string args;    // a piece of the arguments JSON
};

struct Delta {
  std::string reasoning;
  std::string content;
  std::vector<ToolDelta> tools;
  bool empty() const { return reasoning.empty() && content.empty() && tools.empty(); }
};

struct MarkerIds {
  int32_t think_open = -1, think_close = -1, tool_open = -1, tool_close = -1;
};

class OutputParser {
 public:
  // thinking: the prompt ended inside <think> (the output starts as reasoning). tools: the request's tools
  // (empty: tool calls are not parsed, their tokens pass through as text).
  OutputParser(bool thinking, const json& tools, MarkerIds ids, std::string call_prefix = "call_");

  // One generated token: its id and its text (complete UTF-8; special tokens other than the markers arrive with
  // their content and pass as text).
  void push(int32_t id, const std::string& text, Delta& out);
  // End of output: flush what was held back and close an open call.
  void finish(Delta& out);

  int tool_calls() const { return calls_; }
  // The whole message, for non-streaming responses
  const std::string& reasoning() const { return all_reasoning_; }
  const std::string& content() const { return all_content_; }
  const json& message_tool_calls() const { return all_calls_; }

 private:
  enum class State { Reasoning, Content, Call };
  void text_in_call(const std::string& text, Delta& out);
  void parse_call(Delta& out, bool final);
  void emit_content(const std::string& s, Delta& out, bool final);
  void close_call(Delta& out);
  std::string param_type(const std::string& name) const;
  void emit_args(const std::string& piece, Delta& out);

  State state_;
  json tools_;
  bool tools_on_;
  MarkerIds ids_;
  std::string prefix_;
  int calls_ = 0;
  std::string pending_ws_;    // whitespace content held until something else shows up

  // the call being parsed
  enum class Stage { Name, Params, Value, Done };
  Stage stage_ = Stage::Name;
  std::string buf_;           // call text not consumed yet
  std::string fn_;
  int n_params_ = 0;
  std::string param_, ptype_;
  bool value_string_ = false, value_started_ = false;
  std::string value_;         // non-string value (or the start of a string, until its first char is known)
  std::string call_args_;     // the arguments JSON so far
  std::string call_id_;

  std::string all_reasoning_, all_content_;
  json all_calls_ = json::array();
};

// The JSON text for a parameter value of the given schema type (as SGLang's qwen3_coder detector converts it).
std::string convert_value(const std::string& raw, const std::string& type);
std::string json_escape(const std::string& s);

}  // namespace xtg
