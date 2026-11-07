#include "xtg/sampler.h"

#include <algorithm>
#include <cmath>

namespace xtg {

int argmax(const float* logits, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (logits[i] > logits[best]) best = i;
  return best;
}

static float log_sum_exp(const float* logits, int n) {
  float mx = logits[0];
  for (int i = 1; i < n; ++i) mx = std::max(mx, logits[i]);
  double s = 0.0;
  for (int i = 0; i < n; ++i) s += std::exp(double(logits[i] - mx));
  return mx + float(std::log(s));
}

float logprob(const float* logits, int n, int token) { return logits[token] - log_sum_exp(logits, n); }

static bool more_likely(const std::pair<float, int>& a, const std::pair<float, int>& b) {
  return a.first > b.first || (a.first == b.first && a.second < b.second);
}

void top_logprobs(const float* logits, int n, int k, std::vector<std::pair<float, int>>& out) {
  float lse = log_sum_exp(logits, n);
  out.clear();
  k = std::min(k, n);
  if (k <= 0) return;
  std::vector<std::pair<float, int>> all(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) all[size_t(i)] = {logits[i], i};
  std::partial_sort(all.begin(), all.begin() + k, all.end(), more_likely);
  for (int i = 0; i < k; ++i) out.push_back({all[size_t(i)].first - lse, all[size_t(i)].second});
}

int Sampler::sample(const float* logits, int n, const SamplingParams& p) {
  if (p.temperature <= 0.f) return argmax(logits, n);
  float mx = logits[0];
  for (int i = 1; i < n; ++i) mx = std::max(mx, logits[i]);
  const float inv_t = 1.f / p.temperature;
  cand_.resize(size_t(n));
  double total = 0.0;
  for (int i = 0; i < n; ++i) {
    float w = std::exp((logits[i] - mx) * inv_t);
    cand_[size_t(i)] = {w, i};
    total += w;
  }
  // Candidates most likely first, as far as the filters need: top_k, else a sorted head that covers top_p
  // (a full sort of 131K entries per token would cost a phone ~10 ms).
  size_t keep = size_t(n);
  if (p.top_k > 0 && p.top_k < n) {
    keep = size_t(p.top_k);
    std::partial_sort(cand_.begin(), cand_.begin() + long(keep), cand_.end(), more_likely);
    total = 0.0;
    for (size_t i = 0; i < keep; ++i) total += cand_[i].first;
  } else if (p.top_p < 1.f || p.min_p > 0.f) {
    size_t head = std::min<size_t>(size_t(n), 1024);
    std::partial_sort(cand_.begin(), cand_.begin() + long(head), cand_.end(), more_likely);
    double cum = 0.0;
    for (size_t i = 0; i < head; ++i) cum += cand_[i].first;
    if (head < size_t(n) && p.top_p < 1.f && cum < total * double(p.top_p))
      std::sort(cand_.begin() + long(head), cand_.end(), more_likely);
  }
  if (p.min_p > 0.f) {
    float floor = p.min_p * cand_[0].first;
    size_t k = 0;
    double t = 0.0;
    while (k < keep && cand_[k].first >= floor) t += cand_[k++].first;
    keep = std::max<size_t>(1, k);
    total = k ? t : double(cand_[0].first);
  }
  if (p.top_p < 1.f) {
    double target = total * double(p.top_p), cum = 0.0;
    size_t k = 0;
    while (k < keep) {
      cum += cand_[k].first;
      ++k;
      if (cum >= target) break;
    }
    keep = std::max<size_t>(1, k);
    total = cum;
  }
  std::uniform_real_distribution<double> uni(0.0, total);
  double r = uni(rng_), cum = 0.0;
  for (size_t i = 0; i < keep; ++i) {
    cum += cand_[i].first;
    if (r < cum) return cand_[i].second;
  }
  return cand_[keep - 1].second;
}

}  // namespace xtg
