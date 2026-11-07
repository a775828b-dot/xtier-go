// xtg-cli: the engine without the server, for checks and measurements.
//
//   xtg-cli tokenize --tokenizer tokenizer.json --in TEXT_FILE            ids as JSON (special tokens parsed)
//   xtg-cli render   --model DIR --request REQUEST.json                   the chat prompt the server would build
//   xtg-cli forward  --model DIR --ids IDS.json [--opt k=v ...] [--logits OUT.f32] [--threads N]
//                    prefill the ids; prints per position the next token's log probability and the argmax;
//                    --logits writes every row's logits (float32, rows x vocab)
//   xtg-cli generate --model DIR (--prompt TEXT | --ids IDS.json) [--max N] [--temp T] [--opt k=v ...]
//                    greedy (or sampled) continuation with prefill / decode speed
//   xtg-cli estimate --model DIR [--opt k=v ...]                          memory the options would take
#include <cstdio>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "chat.h"
#include "tokenizer.h"
#include "xtg/common.h"
#include "xtg/json.h"
#include "xtg/model.h"
#include "xtg/sampler.h"

using namespace xtg;

namespace {

struct Args {
  std::string cmd;
  std::map<std::string, std::string> kv;
  json opts = json::object();
};

Args parse(int argc, char** argv) {
  Args a;
  XTG_CHECK(argc >= 2, "usage: xtg-cli tokenize|render|forward|generate|estimate ... (see the source header)");
  a.cmd = argv[1];
  for (int i = 2; i < argc; ++i) {
    std::string k = argv[i];
    XTG_CHECK(k.rfind("--", 0) == 0 && i + 1 < argc, "bad argument %s", k.c_str());
    std::string v = argv[++i];
    if (k == "--opt") {
      size_t eq = v.find('=');
      XTG_CHECK(eq != std::string::npos, "--opt wants key=value, got %s", v.c_str());
      a.opts[v.substr(0, eq)] = v.substr(eq + 1);
    } else {
      a.kv[k.substr(2)] = v;
    }
  }
  return a;
}

std::string read_file(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  XTG_CHECK(f.good(), "cannot read %s", path.c_str());
  std::stringstream s;
  s << f.rdbuf();
  return s.str();
}

std::string need(const Args& a, const char* k) {
  auto it = a.kv.find(k);
  XTG_CHECK(it != a.kv.end(), "missing --%s", k);
  return it->second;
}

int threads(const Args& a) {
  auto it = a.kv.find("threads");
  if (it != a.kv.end()) return std::stoi(it->second);
  return std::max(1u, std::thread::hardware_concurrency());
}

std::vector<int32_t> read_ids(const std::string& path) { return json::parse(read_file(path)).get<std::vector<int32_t>>(); }

}  // namespace

int main(int argc, char** argv) {
  try {
    Args a = parse(argc, argv);
    register_builtin_packs();
    if (a.cmd == "tokenize") {
      Tokenizer tok(need(a, "tokenizer"));
      std::cout << json(tok.encode(read_file(need(a, "in")))).dump() << "\n";
      return 0;
    }
    if (a.cmd == "render") {
      ChatTemplate tmpl(need(a, "model"));
      json req = json::parse(read_file(need(a, "request")));
      std::cout << tmpl.render(req.at("messages"), req.value("tools", json::array()), true,
                               req.value("chat_template_kwargs", json::object()));
      return 0;
    }
    if (a.cmd == "estimate") {
      std::cout << estimate_memory(need(a, "model"), a.opts).dump(1) << "\n";
      return 0;
    }
    std::string dir = need(a, "model");
    ThreadPool pool(threads(a));
    auto model = load_model(dir, a.opts, pool);
    auto session = model->new_session();
    const int vocab = model->info().vocab;
    if (a.cmd == "forward") {
      std::vector<int32_t> ids = read_ids(need(a, "ids"));
      std::ofstream logits_out;
      if (a.kv.count("logits")) logits_out.open(a.kv["logits"], std::ios::binary);
      json lp = json::array(), am = json::array();
      double t0 = now_s();
      session->forward(ids.data(), int(ids.size()), Logits::All, [&](int row, const float* lg) {
        if (logits_out.is_open()) logits_out.write(reinterpret_cast<const char*>(lg), std::streamsize(vocab) * 4);
        am.push_back(argmax(lg, vocab));
        lp.push_back(row + 1 < int(ids.size()) ? json(logprob(lg, vocab, ids[size_t(row) + 1])) : json(nullptr));
      });
      double dt = now_s() - t0;
      std::cout << json{{"argmax", am}, {"logprob_next", lp}, {"seconds", dt}, {"tokens", ids.size()}}.dump() << "\n";
      return 0;
    }
    if (a.cmd == "generate") {
      std::vector<int32_t> ids;
      std::unique_ptr<Tokenizer> tok;
      if (a.kv.count("prompt")) {
        tok = std::make_unique<Tokenizer>(dir + "/tokenizer.json");
        ids = tok->encode(a.kv["prompt"]);
      } else {
        ids = read_ids(need(a, "ids"));
      }
      int max_new = a.kv.count("max") ? std::stoi(a.kv["max"]) : 64;
      SamplingParams sp;
      sp.temperature = a.kv.count("temp") ? std::stof(a.kv["temp"]) : 0.f;
      Sampler sampler(1234);
      int next = -1;
      double t0 = now_s();
      session->forward(ids.data(), int(ids.size()), Logits::Last,
                       [&](int, const float* lg) { next = sampler.sample(lg, vocab, sp); });
      double t1 = now_s();
      std::vector<int32_t> out;
      const auto& eos = model->info().eos_ids;
      for (int i = 0; i < max_new; ++i) {
        out.push_back(next);
        if (std::find(eos.begin(), eos.end(), next) != eos.end()) break;
        int32_t t = next;
        session->forward(&t, 1, Logits::Last, [&](int, const float* lg) { next = sampler.sample(lg, vocab, sp); });
      }
      double t2 = now_s();
      if (!tok) tok = std::make_unique<Tokenizer>(dir + "/tokenizer.json");
      std::cout << json{{"ids", out},
                        {"text", tok->decode(out, false)},
                        {"prompt_tokens", ids.size()},
                        {"prefill_tok_s", double(ids.size()) / (t1 - t0)},
                        {"decode_tok_s", out.size() > 1 ? double(out.size() - 1) / (t2 - t1) : 0.0}}
                       .dump()
                << "\n";
      return 0;
    }
    fail("unknown command %s", a.cmd.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "xtg-cli: %s\n", e.what());
    return 1;
  }
}
