// Kernel checks without a model: number formats, NVFP4 matmul against a naive decode, attention against a naive
// softmax (full, sliding window, cache + new keys, split keys), the KV ring, the sampler. On aarch64 the NEON
// kernels are compared with the portable ones too. Prints FAIL lines and exits 1 on any failure.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "xtg/fp.h"
#include "xtg/kernels.h"
#include "xtg/kv.h"
#include "xtg/sampler.h"
#include "xtg/threadpool.h"

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

static std::mt19937 rng(42);
static float uni(float a, float b) { return std::uniform_real_distribution<float>(a, b)(rng); }

static void test_fp8() {
  CHECK(e4m3_to_f32(0x38) == 1.f, "e4m3 0x38");
  CHECK(e4m3_to_f32(0x7e) == 448.f, "e4m3 0x7e");
  CHECK(e4m3_to_f32(0x01) == std::ldexp(1.f, -9), "e4m3 smallest subnormal");
  CHECK(std::isnan(e4m3_to_f32(0x7f)), "e4m3 NaN");
  for (int c = 0; c < 256; ++c) {
    if ((c & 0x7f) == 0x7f) continue;
    CHECK(f32_to_e4m3(e4m3_to_f32(uint8_t(c))) == c, "e4m3 round trip %d", c);
  }
  CHECK(f32_to_e4m3(1000.f) == 0x7e && f32_to_e4m3(-1000.f) == 0xfe, "e4m3 saturates");
  CHECK(f32_to_bf16(1.f) == 0x3f80 && bf16_to_f32(0x3f80) == 1.f, "bf16");
}

static Nvfp4Weight random_weight(int n, int k, std::vector<uint8_t>& codes, std::vector<uint8_t>& scales) {
  codes.resize(size_t(n) * size_t(k / 2));
  scales.resize(size_t(n) * size_t(k / 16));
  for (auto& c : codes) c = uint8_t(rng() & 0xff);
  for (auto& s : scales) s = uint8_t(0x20 + (rng() % 0x30));   // positive normal E4M3 values
  Nvfp4Weight w;
  w.codes = codes.data();
  w.scales = scales.data();
  w.global = 0.37f;
  w.n_out = n;
  w.k = k;
  return w;
}

static void test_nvfp4(ThreadPool& pool) {
  const int n = 70, k = 256;
  std::vector<uint8_t> c1, s1, c2, s2;
  Nvfp4Weight a = random_weight(n, k, c1, s1), b = random_weight(33, k, c2, s2);
  for (int m : {1, 2, 7}) {
    std::vector<float> x(size_t(m) * k), ya(size_t(m) * n), yb(size_t(m) * 33);
    for (auto& v : x) v = uni(-1, 1);
    matmul_nvfp4(pool, {&a, &b}, x.data(), m, {ya.data(), yb.data()});
    std::vector<float> row(k);
    // error against a double-precision dot, in units of the outputs' RMS (FP32 summation order differs between
    // compilers and kernels; an output near zero must not count as a large relative error)
    double worst = 0, sq = 0;
    int count = 0;
    std::vector<double> err;
    for (int which = 0; which < 2; ++which) {
      const Nvfp4Weight& w = which ? b : a;
      float* y = which ? yb.data() : ya.data();
      for (int i = 0; i < m; ++i)
        for (int r = 0; r < w.n_out; ++r) {
          nvfp4_row(w, r, row.data());
          double ref = 0;
          for (int j = 0; j < k; ++j) ref += double(row[size_t(j)]) * x[size_t(i) * k + j];
          err.push_back(std::fabs(ref - y[size_t(i) * w.n_out + r]));
          sq += ref * ref;
          ++count;
        }
    }
    double rms = std::sqrt(sq / count);
    for (double e : err) worst = std::max(worst, e / rms);
    CHECK(worst < 2e-5, "nvfp4 matmul m=%d: error %g of the RMS output", m, worst);
  }
#if defined(__aarch64__)
  std::vector<float> x(k), y0(n), y1(n);
  for (auto& v : x) v = uni(-1, 1);
  set_kernel_set(KernelSet::Portable);
  matmul_nvfp4(pool, {&a}, x.data(), 1, {y0.data()});
  set_kernel_set(KernelSet::Neon);
  matmul_nvfp4(pool, {&a}, x.data(), 1, {y1.data()});
  double worst = 0, sq = 0;
  for (int r = 0; r < n; ++r) sq += double(y0[r]) * y0[r];
  for (int r = 0; r < n; ++r) worst = std::max(worst, double(std::fabs(y0[r] - y1[r])) / std::sqrt(sq / n));
  CHECK(worst < 2e-5, "neon vs portable nvfp4 gemv: %g of the RMS output", worst);
  std::printf("neon nvfp4 gemv checked\n");
#endif
}

// naive attention over explicit K / V rows (positions 0 .. pos0 + n_q - 1)
static void naive_attention(const AttnArgs& a, const std::vector<float>& K, const std::vector<float>& V, std::vector<float>& out) {
  const int g = a.heads / a.kv_heads, d = a.head_dim;
  out.assign(size_t(a.n_q) * a.heads * d, 0.f);
  for (int i = 0; i < a.n_q; ++i) {
    int64_t t = a.pos0 + i, lo = a.window > 0 ? std::max<int64_t>(0, t - a.window + 1) : 0;
    for (int h = 0; h < a.heads; ++h) {
      int kh = h / g;
      const float* q = a.q + (size_t(i) * a.heads + h) * d;
      std::vector<double> s;
      double mx = -1e300;
      for (int64_t p = lo; p <= t; ++p) {
        const float* kr = K.data() + (size_t(p) * a.kv_heads + kh) * d;
        double dot = 0;
        for (int e = 0; e < d; ++e) dot += double(q[e]) * kr[e];
        s.push_back(dot * a.scale);
        mx = std::max(mx, s.back());
      }
      double sum = 0;
      for (double& v : s) sum += (v = std::exp(v - mx));
      float* o = out.data() + (size_t(i) * a.heads + h) * d;
      for (int64_t p = lo; p <= t; ++p) {
        const float* vr = V.data() + (size_t(p) * a.kv_heads + kh) * d;
        for (int e = 0; e < d; ++e) o[e] += float(s[size_t(p - lo)] / sum * vr[e]);
      }
    }
  }
}

static void test_attention(ThreadPool& pool) {
  const int heads = 8, kvh = 2, d = 64;
  struct Case {
    int64_t cached;
    int n_q, window;
    KvDtype dt;
  };
  for (Case cs : {Case{300, 1, 0, KvDtype::BF16}, Case{5000, 1, 0, KvDtype::BF16}, Case{300, 1, 64, KvDtype::BF16},
                  Case{100, 37, 0, KvDtype::BF16}, Case{100, 37, 16, KvDtype::FP8}, Case{0, 20, 0, KvDtype::FP8},
                  Case{900, 3, 128, KvDtype::FP8}}) {
    int64_t total = cs.cached + cs.n_q;
    std::vector<float> K(size_t(total) * kvh * d), V(K.size()), q(size_t(cs.n_q) * heads * d);
    for (auto& v : K) v = uni(-1, 1);
    for (auto& v : V) v = uni(-1, 1);
    for (auto& v : q) v = uni(-1, 1);
    KvCache cache(kvh, d, total + 8, cs.window > 0 ? cs.window : 0, cs.dt);
    // the cache holds positions < cached; the reference sees exactly what the cache gives back
    for (int64_t p = 0; p < cs.cached; p += 50) {
      int n = int(std::min<int64_t>(50, cs.cached - p));
      cache.store(p, n, K.data() + size_t(p) * kvh * d, V.data() + size_t(p) * kvh * d);
    }
    std::vector<float> Kc = K, Vc = V;
    int64_t lo = cs.window > 0 ? std::max<int64_t>(0, cs.cached - cs.window) : 0;
    for (int h = 0; h < kvh; ++h) {
      std::vector<float> kb(d), vb(d);
      for (int64_t p = lo; p < cs.cached; ++p) {
        cache.load(h, p, p + 1, kb.data(), vb.data());
        std::copy(kb.begin(), kb.end(), Kc.begin() + (size_t(p) * kvh + h) * d);
        std::copy(vb.begin(), vb.end(), Vc.begin() + (size_t(p) * kvh + h) * d);
      }
    }
    AttnArgs a;
    a.n_q = cs.n_q;
    a.pos0 = cs.cached;
    a.heads = heads;
    a.kv_heads = kvh;
    a.head_dim = d;
    a.window = cs.window;
    a.scale = 0.7f;
    a.q = q.data();
    a.k = K.data() + size_t(cs.cached) * kvh * d;
    a.v = V.data() + size_t(cs.cached) * kvh * d;
    a.cache = &cache;
    std::vector<float> out(q.size()), ref;
    a.out = out.data();
    attention(pool, a);
    naive_attention(a, Kc, Vc, ref);
    double worst = 0;
    for (size_t i = 0; i < out.size(); ++i) worst = std::max(worst, double(std::fabs(out[i] - ref[i])));
    CHECK(worst < 2e-5, "attention cached=%lld n_q=%d window=%d %s: max error %g", (long long)cs.cached, cs.n_q,
          cs.window, kv_dtype_name(cs.dt), worst);
  }
  // BF16 / FP8 storage keeps values close to what was stored
  KvCache ring(1, 64, 0, 64, KvDtype::FP8);
  std::vector<float> k(3000 * 64), v(3000 * 64), kb(64 * 64), vb(64 * 64);
  for (auto& x : k) x = uni(-0.3f, 0.3f);
  for (auto& x : v) x = uni(-2, 2);
  ring.store(0, 3000, k.data(), v.data());
  ring.load(0, 2936, 3000, kb.data(), vb.data());
  double worst = 0;
  for (size_t i = 0; i < kb.size(); ++i) worst = std::max(worst, double(std::fabs(kb[i] - k[2936 * 64 + i])) / 0.3);
  CHECK(worst < 0.07, "fp8 ring keeps the last 64 positions (error %g)", worst);
}

static void test_sampler() {
  std::vector<float> lg(1000);
  for (auto& x : lg) x = uni(-5, 5);
  lg[123] = 20.f;
  Sampler s(7);
  SamplingParams greedy;
  greedy.temperature = 0.f;
  CHECK(s.sample(lg.data(), 1000, greedy) == 123, "greedy");
  SamplingParams topk;
  topk.top_k = 1;
  CHECK(s.sample(lg.data(), 1000, topk) == 123, "top_k 1");
  SamplingParams topp;
  topp.top_p = 0.5f;
  CHECK(s.sample(lg.data(), 1000, topp) == 123, "top_p 0.5 with one dominant token");
  CHECK(std::fabs(logprob(lg.data(), 1000, 123)) < 1e-3, "logprob of the dominant token");
}

static void test_pool(ThreadPool& pool) {
  std::vector<int> hits(10000, 0);
  for (int rep = 0; rep < 50; ++rep)
    pool.parallel_for(10000, 7, [&](int64_t b, int64_t e, int) {
      for (int64_t i = b; i < e; ++i) hits[size_t(i)]++;
    });
  bool ok = true;
  for (int h : hits) ok &= h == 50;
  CHECK(ok, "parallel_for covers every index once per call");
}

int main() {
  ThreadPool pool(4);
  test_fp8();
  test_pool(pool);
  test_nvfp4(pool);
  test_attention(pool);
  test_sampler();
  if (g_fail)
    std::printf("%d FAILED\n", g_fail);
  else
    std::printf("all kernel checks passed (%s kernels)\n", kernel_set_name());
  return g_fail ? 1 : 0;
}
