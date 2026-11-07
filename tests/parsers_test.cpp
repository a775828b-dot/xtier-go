// Output parser checks: scripted token streams (markers as ids, text cut at awkward places) -> the deltas
// concatenated must equal the whole message, and the whole message must be what the model meant.
#include <cstdio>
#include <string>
#include <vector>

#include "parsers.h"

using namespace xtg;

static int g_fail = 0;
#define CHECK(cond, ...)                \
  do {                                  \
    if (!(cond)) {                      \
      std::printf("FAIL: " __VA_ARGS__); \
      std::printf("\n");                \
      ++g_fail;                         \
    }                                   \
  } while (0)

static const MarkerIds kIds{12, 13, 14, 15};
enum : int32_t { THINK = 12, END_THINK = 13, CALL = 14, END_CALL = 15, TEXT = 100 };

struct Step {
  int32_t id;
  std::string text;
};

struct Result {
  std::string reasoning, content;
  std::vector<std::string> names, args;   // per call, from the deltas
  json calls;                              // the parser's whole message
  int n_deltas = 0;
};

static Result run(bool thinking, const json& tools, const std::vector<Step>& steps) {
  OutputParser p(thinking, tools, kIds);
  Result r;
  auto take = [&](const Delta& d) {
    r.reasoning += d.reasoning;
    r.content += d.content;
    for (const auto& t : d.tools) {
      if (int(r.args.size()) <= t.index) {
        r.args.resize(size_t(t.index) + 1);
        r.names.resize(size_t(t.index) + 1);
      }
      if (!t.name.empty()) r.names[size_t(t.index)] = t.name;
      r.args[size_t(t.index)] += t.args;
      ++r.n_deltas;
    }
  };
  for (const auto& s : steps) {
    Delta d;
    p.push(s.id, s.text, d);
    take(d);
  }
  Delta d;
  p.finish(d);
  take(d);
  r.calls = p.message_tool_calls();
  CHECK(r.reasoning == p.reasoning() && r.content == p.content(), "deltas add up to the message");
  for (size_t i = 0; i < r.args.size() && i < r.calls.size(); ++i)
    CHECK(r.args[i] == r.calls[i]["function"]["arguments"].get<std::string>(), "streamed arguments of call %zu", i);
  return r;
}

// text cut into pieces of n bytes (as tokens would)
static std::vector<Step> pieces(const std::string& s, size_t n) {
  std::vector<Step> out;
  for (size_t i = 0; i < s.size(); i += n) out.push_back({TEXT, s.substr(i, n)});
  return out;
}

static std::vector<Step> cat(std::vector<std::vector<Step>> parts) {
  std::vector<Step> out;
  for (auto& p : parts) out.insert(out.end(), p.begin(), p.end());
  return out;
}

static const json kTools = json::parse(R"([
 {"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{
   "path":{"type":"string"},"max_lines":{"type":"integer"},"follow":{"type":"boolean"},
   "opts":{"type":"object"},"ratio":{"type":"number"}}}}},
 {"type":"function","function":{"name":"write_file","parameters":{"type":"object","properties":{
   "path":{"type":"string"},"content":{"type":"string"}}}}}])");

int main() {
  {  // reasoning, then content
    auto r = run(true, json::array(), cat({pieces("Let me think about it.\nOK.", 3), {{END_THINK, "</think>"}},
                                           pieces("\n\nThe answer is 42.", 4)}));
    CHECK(r.reasoning == "Let me think about it.\nOK.", "reasoning: %s", r.reasoning.c_str());
    CHECK(r.content == "\n\nThe answer is 42.", "content: %s", r.content.c_str());
  }
  for (size_t cut : {1, 2, 5, 13, 100}) {  // one call with typed parameters, cut anywhere
    std::string call =
        "\n<function=read_file>\n<parameter=path>\nsrc/main.cpp\n</parameter>\n<parameter=max_lines>\n20\n</parameter>\n"
        "<parameter=follow>\nTrue\n</parameter>\n<parameter=opts>\n{\"a\": [1, 2]}\n</parameter>\n"
        "<parameter=ratio>\n0.5\n</parameter>\n</function>\n";
    auto r = run(true, kTools, cat({pieces("I need the file.", 4), {{END_THINK, "</think>"}}, {{TEXT, "\n\n"}},
                                    {{CALL, "<tool_call>"}}, pieces(call, cut), {{END_CALL, "</tool_call>"}}}));
    CHECK(r.calls.size() == 1 && r.names.size() == 1 && r.names[0] == "read_file", "call name (cut %zu)", cut);
    json args = json::parse(r.calls.size() ? r.calls[0]["function"]["arguments"].get<std::string>() : "{}");
    CHECK(args == json::parse(R"({"path":"src/main.cpp","max_lines":20,"follow":true,"opts":{"a":[1,2]},"ratio":0.5})"),
          "typed arguments (cut %zu): %s", cut, args.dump().c_str());
    CHECK(r.content.empty(), "whitespace around a call is not content (cut %zu): [%s]", cut, r.content.c_str());
  }
  {  // a call inside the reasoning still counts; a call left open at the end is kept
    auto r = run(true, kTools, cat({pieces("I'll read it.", 5), {{CALL, "<tool_call>"}},
                                    pieces("\n<function=read_file>\n<parameter=path>\na.txt", 4)}));
    CHECK(r.reasoning == "I'll read it.", "reasoning before a call");
    CHECK(r.calls.size() == 1 && json::parse(r.calls[0]["function"]["arguments"].get<std::string>()) ==
                                     json::parse(R"({"path":"a.txt"})"),
          "unclosed call kept: %s", r.calls.dump().c_str());
  }
  {  // a long string value streams before the call closes, with escapes
    std::string body;
    for (int i = 0; i < 400; ++i) body += "line " + std::to_string(i) + " \"quoted\" \\ tab\there\n";
    body += "末尾中文";
    auto steps = cat({{{CALL, "<tool_call>"}}, pieces("\n<function=write_file>\n<parameter=path>\nout.txt\n</parameter>\n"
                                                       "<parameter=content>\n" + body + "\n</parameter>\n</function>\n", 7)});
    OutputParser p(false, kTools, kIds);
    size_t streamed_before_close = 0;
    std::string args;
    for (const auto& s : steps) {
      Delta d;
      p.push(s.id, s.text, d);
      for (const auto& t : d.tools) args += t.args;
    }
    streamed_before_close = args.size();
    Delta d;
    p.push(END_CALL, "</tool_call>", d);
    p.finish(d);
    for (const auto& t : d.tools) args += t.args;
    CHECK(streamed_before_close > body.size(), "the value streamed before the close (%zu bytes)", streamed_before_close);
    json a = json::parse(args, nullptr, false);
    CHECK(!a.is_discarded() && a["content"] == body && a["path"] == "out.txt", "long value round trip");
  }
  {  // two calls, then whitespace: finish_reason material
    std::string c1 = "\n<function=read_file>\n<parameter=path>\na\n</parameter>\n</function>\n";
    std::string c2 = "\n<function=read_file>\n<parameter=path>\nb\n</parameter>\n<parameter=unknown>\n7\n</parameter>\n</function>\n";
    auto r = run(false, kTools, cat({{{CALL, ""}}, pieces(c1, 6), {{END_CALL, ""}}, {{TEXT, "\n"}}, {{CALL, ""}},
                                     pieces(c2, 9), {{END_CALL, ""}}, {{TEXT, "\n"}}}));
    CHECK(r.calls.size() == 2, "two calls");
    CHECK(r.calls.size() == 2 && json::parse(r.calls[1]["function"]["arguments"].get<std::string>()) ==
                                     json::parse(R"({"path":"b","unknown":"7"})"),
          "a parameter the schema does not list stays a string");
    CHECK(r.content.empty(), "trailing whitespace dropped after calls");
    CHECK(r.calls.size() == 2 && r.calls[0]["id"] != r.calls[1]["id"], "distinct call ids");
  }
  {  // no tools in the request: the markers are plain text
    auto r = run(false, json::array(), {{CALL, "<tool_call>"}, {TEXT, "x"}, {END_CALL, "</tool_call>"}});
    CHECK(r.content == "<tool_call>x</tool_call>" && r.calls.empty(), "markers as text without tools: %s", r.content.c_str());
  }
  {  // not a call after all: the text comes back as content
    auto r = run(false, kTools, {{CALL, "<tool_call>"}, {TEXT, "just text"}, {END_CALL, "</tool_call>"}});
    CHECK(r.calls.empty() && r.content == "<tool_call>just text", "malformed call: %s", r.content.c_str());
  }
  {  // a call without parameters
    auto r = run(false, kTools, cat({{{CALL, ""}}, pieces("\n<function=read_file>\n</function>\n", 3), {{END_CALL, ""}}}));
    CHECK(r.calls.size() == 1 && r.calls[0]["function"]["arguments"] == "{}", "no parameters");
  }
  CHECK(convert_value("12", "integer") == "12" && convert_value("x", "integer") == "\"x\"", "integer conversion");
  CHECK(convert_value("[1, 2]", "array") == "[1,2]" && convert_value("not json", "object") == "\"not json\"",
        "array / object conversion");
  if (g_fail)
    std::printf("%d FAILED\n", g_fail);
  else
    std::printf("all parser checks passed\n");
  return g_fail ? 1 : 0;
}
