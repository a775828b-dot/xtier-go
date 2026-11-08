// npu_test PTE TOKENS.bin N [REWIND]: libxtg_npu on its own. Prefill the uint64 tokens (qnn_llama_runner's
// --tokenized_prompt format), then N greedy decode steps; rates and the generated ids. With REWIND, go back that
// many tokens after the first run and generate again from there (the two runs must agree).
// npu_test PTE TOKENS.bin score OUT.txt: the prompt's own next-token NLL and argmax at every position (one line
// each, "argmax nll"), to compare with reference/dory_ref.py on the server.
// npu_test PTE TOKENS.bin verify N: N greedy tokens one at a time, then the same tokens again as verify steps of up
// to decode_ar rows (what a speculative decoder sends): the argmax of every row must be the next generated token.
// Then the time of one step of k = 1, 2, 4, .. decode_ar tokens (a forward and a rewind).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "xtg_npu.h"

static double now() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// resident memory of this process (VmRSS), GiB
static double rss_gib() {
  std::ifstream f("/proc/self/status");
  std::string line;
  while (std::getline(f, line))
    if (line.rfind("VmRSS:", 0) == 0) return atof(line.c_str() + 6) / (1024.0 * 1024.0);
  return 0;
}

struct Argmax {
  int vocab;
  int best = -1;
};

static void on_logits(int, const float* lg, void* u) {
  auto* a = static_cast<Argmax*>(u);
  int b = 0;
  for (int j = 1; j < a->vocab; ++j)
    if (lg[j] > lg[b]) b = j;
  a->best = b;
}

static std::vector<int> generate(xtg_npu* h, int first, int n, int vocab, double* rate) {
  std::vector<int> out{first};
  char err[512];
  Argmax a{vocab};
  double t0 = now();
  for (int i = 1; i < n; ++i) {
    int32_t t = out.back();
    if (xtg_npu_forward(h, &t, 1, 1, on_logits, &a, err, sizeof(err)) != 0) {
      fprintf(stderr, "decode: %s\n", err);
      break;
    }
    out.push_back(a.best);
  }
  *rate = (out.size() - 1) / (now() - t0);
  return out;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: npu_test PTE TOKENS.bin N [REWIND]\n");
    return 2;
  }
  std::ifstream f(argv[2], std::ios::binary | std::ios::ate);
  std::vector<uint64_t> raw(size_t(f.tellg()) / 8);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(raw.data()), std::streamsize(raw.size() * 8));
  std::vector<int32_t> prompt(raw.begin(), raw.end());
  int n = atoi(argv[3]), rew = argc > 4 ? atoi(argv[4]) : 0;
  char err[512], info[1024];
  double t0 = now();
  const char* opts = getenv("XTG_NPU_OPTIONS");   // the plugin's options JSON, e.g. {"decode_rows": -1}
  xtg_npu* h = xtg_npu_open(argv[1], opts ? opts : "", err, sizeof(err));
  if (!h) {
    fprintf(stderr, "open: %s\n", err);
    return 1;
  }
  xtg_npu_info(h, info, sizeof(info));
  printf("loaded in %.1f s, resident %.2f GiB: %s\n", now() - t0, rss_gib(), info);
  int vocab = atoi(strstr(info, "\"vocab\":") + 8);
  if (std::string(argv[3]) == "score") {
    struct Score {
      int vocab;
      const std::vector<int32_t>* ids;
      FILE* out;
      int off = 0;   // the position of this call's first row
      double nll = 0;
      int rows = 0;
    } s{vocab, &prompt, fopen(argc > 4 ? argv[4] : "score.txt", "w")};
    t0 = now();
    // XTG_SCORE_STEP=n: n tokens per call (1: the decode graph), default all at once (prefill chunks)
    const char* step_env = getenv("XTG_SCORE_STEP");
    const int step = step_env && atoi(step_env) > 0 ? atoi(step_env) : int(prompt.size());
    auto on_score = [](int r, const float* lg, void* u) {
      auto* s = static_cast<Score*>(u);
      const int row = s->off + r;
      int b = 0;
      float mx = lg[0];
      for (int j = 1; j < s->vocab; ++j)
        if (lg[j] > mx) mx = lg[j], b = j;
      double z = 0;
      for (int j = 0; j < s->vocab; ++j) z += std::exp(double(lg[j] - mx));
      double nll = 0;
      if (row + 1 < int(s->ids->size())) {
        nll = -(double(lg[(*s->ids)[size_t(row) + 1]] - mx) - std::log(z));
        s->nll += nll;
        s->rows++;
      }
      fprintf(s->out, "%d %.6f\n", b, nll);
    };
    int rc = 0;
    for (s.off = 0; rc == 0 && s.off < int(prompt.size()); s.off += step)
      rc = xtg_npu_forward(h, prompt.data() + s.off, std::min(step, int(prompt.size()) - s.off), 2, on_score, &s, err,
                           sizeof(err));
    fclose(s.out);
    if (rc != 0) {
      fprintf(stderr, "score: %s\n", err);
      return 1;
    }
    printf("scored %zu positions in %.1f s: mean next-token NLL %.4f (PPL %.3f)\n", prompt.size(), now() - t0,
           s.nll / s.rows, std::exp(s.nll / s.rows));
    xtg_npu_close(h);
    return 0;
  }
  Argmax a{vocab};
  t0 = now();
  if (xtg_npu_forward(h, prompt.data(), int(prompt.size()), 1, on_logits, &a, err, sizeof(err)) != 0) {
    fprintf(stderr, "prefill: %s\n", err);
    return 1;
  }
  double pre = prompt.size() / (now() - t0);
  double dec = 0;
  if (std::string(argv[3]) == "verify") {
    const int ar = atoi(strstr(info, "\"decode_ar\":") + 12);
    n = argc > 4 ? atoi(argv[4]) : 64;
    std::vector<int> out = generate(h, a.best, n, vocab, &dec);
    printf("prompt %zu tokens: prefill %.1f tok/s; %zu generated one at a time: decode %.2f tok/s\n", prompt.size(),
           pre, out.size(), dec);
    const int64_t base = int64_t(prompt.size());
    if (xtg_npu_rewind(h, base) != 0) {
      fprintf(stderr, "rewind failed\n");
      return 1;
    }
    struct Rows {
      int vocab;
      std::vector<int> best;
    } r{vocab, {}};
    auto on_rows = [](int, const float* lg, void* u) {
      auto* r = static_cast<Rows*>(u);
      int b = 0;
      for (int j = 1; j < r->vocab; ++j)
        if (lg[j] > lg[b]) b = j;
      r->best.push_back(b);
    };
    std::vector<int32_t> ids(out.begin(), out.end() - 1);   // row i predicts out[i + 1]
    t0 = now();
    int steps = 0;
    for (size_t off = 0; off < ids.size(); off += size_t(ar), ++steps)
      if (xtg_npu_forward(h, ids.data() + off, int(std::min(size_t(ar), ids.size() - off)), 2, on_rows, &r, err,
                          sizeof(err)) != 0) {
        fprintf(stderr, "verify: %s\n", err);
        return 1;
      }
    const double vt = now() - t0;
    int same = 0;
    for (size_t i = 0; i < r.best.size(); ++i) same += r.best[i] == out[i + 1];
    printf("verify in steps of %d: %d of %zu rows agree with the one-at-a-time run, %.1f ms per step\n", ar, same,
           r.best.size(), vt * 1e3 / steps);
    for (int k = 1; k <= ar; k *= 2) {
      const int reps = 8;
      double t = 0;
      for (int i = 0; i < reps; ++i) {
        const int64_t at = xtg_npu_pos(h);
        const double s = now();
        if (xtg_npu_forward(h, ids.data(), k, 2, on_rows, &r, err, sizeof(err)) != 0) {
          fprintf(stderr, "step %d: %s\n", k, err);
          return 1;
        }
        t += now() - s;
        xtg_npu_rewind(h, at);
      }
      printf("  %d token(s) per step: %.1f ms\n", k, t * 1e3 / reps);
    }
    xtg_npu_close(h);
    return 0;
  }
  std::vector<int> out = generate(h, a.best, n, vocab, &dec);
  printf("prompt %zu tokens: prefill %.1f tok/s; %zu generated: decode %.2f tok/s\nids:", prompt.size(), pre,
         out.size(), dec);
  for (int t : out) printf(" %d", t);
  printf("\n");
  if (rew > 0) {
    // back rew tokens into the generation, run the same tokens again, then generate: the same ids?
    int64_t base = int64_t(prompt.size()) + int64_t(out.size()) - 1 - rew;
    if (xtg_npu_rewind(h, base) != 0) {
      fprintf(stderr, "rewind failed\n");
      return 1;
    }
    size_t k = size_t(base) - prompt.size();   // the generated tokens kept
    int32_t t = out[k];
    xtg_npu_forward(h, &t, 1, 1, on_logits, &a, err, sizeof(err));
    std::vector<int> again = generate(h, a.best, rew, vocab, &dec);
    int same = 0;
    for (size_t i = 0; i < again.size() && k + 1 + i < out.size(); ++i) same += again[i] == out[k + 1 + i];
    printf("rewind %d: %d of %zu tokens the same, decode %.2f tok/s\n", rew, same, again.size(), dec);
  }
  xtg_npu_close(h);
  return 0;
}
