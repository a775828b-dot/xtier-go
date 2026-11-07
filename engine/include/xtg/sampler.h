// Sampling from one row of logits, and log probabilities for the scoring endpoint.
#pragma once

#include <cstdint>
#include <random>
#include <utility>
#include <vector>

namespace xtg {

struct SamplingParams {
  float temperature = 1.f;   // 0: greedy
  float top_p = 1.f;
  int top_k = 0;             // 0: off
  float min_p = 0.f;
};

class Sampler {
 public:
  explicit Sampler(uint64_t seed) : rng_(seed) {}
  void seed(uint64_t s) { rng_.seed(s); }
  int sample(const float* logits, int n, const SamplingParams& p);

 private:
  std::mt19937_64 rng_;
  std::vector<std::pair<float, int>> cand_;
};

int argmax(const float* logits, int n);
// log softmax of one token, and the k most likely (log probability, token) pairs
float logprob(const float* logits, int n, int token);
void top_logprobs(const float* logits, int n, int k, std::vector<std::pair<float, int>>& out);

}  // namespace xtg
