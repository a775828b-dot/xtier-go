// Portable kernels: plain loops the compiler vectorizes. They define the results; the NEON versions must match
// them (tests/kernels_test.cpp).
#include "xtg/kernels.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <mutex>

#include "xtg/common.h"
#include "xtg/fp.h"

namespace xtg {

#if defined(__aarch64__)
// kernels_neon.cpp
void gemv_nvfp4_neon(const Nvfp4Weight& w, int row0, int row1, const float* x, float* y);
bool neon_available();
static KernelSet g_kernels = neon_available() ? KernelSet::Neon : KernelSet::Portable;
#else
static KernelSet g_kernels = KernelSet::Portable;
#endif

KernelSet kernel_set() { return g_kernels; }
const char* kernel_set_name() { return g_kernels == KernelSet::Neon ? "neon" : "portable"; }
void set_kernel_set(KernelSet k) {
#if defined(__aarch64__)
  g_kernels = k;
#else
  (void)k;
#endif
}

void nvfp4_row(const Nvfp4Weight& w, int n, float* out) {
  const uint8_t* c = w.codes + size_t(n) * size_t(w.k / 2);
  const uint8_t* s = w.scales + size_t(n) * size_t(w.k / 16);
  for (int g = 0; g < w.k / 16; ++g) {
    float sc = e4m3_to_f32(s[g]) * w.global;
    for (int j = 0; j < 8; ++j) {
      uint8_t b = c[g * 8 + j];
      out[g * 16 + 2 * j] = kE2m1[b & 15] * sc;
      out[g * 16 + 2 * j + 1] = kE2m1[b >> 4] * sc;
    }
  }
}

// One output of an NVFP4 GEMV: per 16-input group the E2M1 products are summed, then scaled.
static float dot_nvfp4(const Nvfp4Weight& w, int n, const float* x) {
  const uint8_t* c = w.codes + size_t(n) * size_t(w.k / 2);
  const uint8_t* s = w.scales + size_t(n) * size_t(w.k / 16);
  float acc = 0.f;
  for (int g = 0; g < w.k / 16; ++g) {
    const uint8_t* cg = c + g * 8;
    const float* xg = x + g * 16;
    float part = 0.f;
    for (int j = 0; j < 8; ++j) part += kE2m1[cg[j] & 15] * xg[2 * j] + kE2m1[cg[j] >> 4] * xg[2 * j + 1];
    acc += part * e4m3_to_f32(s[g]);
  }
  return acc * w.global;
}

float dot_f32(const float* a, const float* b, int k) {
  float s = 0.f;
#pragma omp simd reduction(+ : s)
  for (int j = 0; j < k; ++j) s += a[j] * b[j];
  return s;
}

namespace {
struct RowMap {
  std::vector<int> begin;   // global first row of each weight
  int total = 0;
  RowMap(const std::vector<const Nvfp4Weight*>& ws) {
    for (auto* w : ws) {
      begin.push_back(total);
      total += w->n_out;
    }
  }
  // the weight holding global row r
  int which(int r) const { return int(std::upper_bound(begin.begin(), begin.end(), r) - begin.begin()) - 1; }
};
}  // namespace

void matmul_nvfp4(ThreadPool& pool, const std::vector<const Nvfp4Weight*>& ws, const float* x, int m,
                  const std::vector<float*>& ys) {
  XTG_CHECK(ws.size() == ys.size() && !ws.empty(), "matmul_nvfp4: %zu weights, %zu outputs", ws.size(), ys.size());
  const int k = ws[0]->k;
  for (auto* w : ws) XTG_CHECK(w->k == k && k % 16 == 0, "matmul_nvfp4: input widths differ (%d, %d)", w->k, k);
  RowMap map(ws);
  if (m <= 2) {
    // decode: one pass over the weights, rows handed out in blocks
    const int block = 16;
    pool.parallel_for((map.total + block - 1) / block, 1, [&](int64_t b0, int64_t b1, int) {
      for (int64_t b = b0; b < b1; ++b) {
        int r0 = int(b) * block, r1 = std::min(map.total, r0 + block);
        while (r0 < r1) {
          int wi = map.which(r0);
          const Nvfp4Weight& w = *ws[size_t(wi)];
          int lo = r0 - map.begin[size_t(wi)], hi = std::min(w.n_out, r1 - map.begin[size_t(wi)]);
          for (int i = 0; i < m; ++i) {
            float* y = ys[size_t(wi)] + size_t(i) * size_t(w.n_out);
#if defined(__aarch64__)
            if (g_kernels == KernelSet::Neon) {
              gemv_nvfp4_neon(w, lo, hi, x + size_t(i) * size_t(k), y);
              continue;
            }
#endif
            for (int n = lo; n < hi; ++n) y[n] = dot_nvfp4(w, n, x + size_t(i) * size_t(k));
          }
          r0 += hi - lo;
        }
      }
    });
    return;
  }
  // prefill: a block of rows decoded to FP32 once per thread, then every input row against it
  const int block = 32;
  pool.parallel_for((map.total + block - 1) / block, 1, [&](int64_t b0, int64_t b1, int) {
    thread_local std::vector<float> wbuf;
    wbuf.resize(size_t(block) * size_t(k));
    for (int64_t b = b0; b < b1; ++b) {
      int r0 = int(b) * block, r1 = std::min(map.total, r0 + block);
      while (r0 < r1) {
        int wi = map.which(r0);
        const Nvfp4Weight& w = *ws[size_t(wi)];
        int lo = r0 - map.begin[size_t(wi)], hi = std::min(w.n_out, r1 - map.begin[size_t(wi)]);
        for (int n = lo; n < hi; ++n) nvfp4_row(w, n, wbuf.data() + size_t(n - lo) * size_t(k));
        float* y = ys[size_t(wi)];
        for (int i = 0; i < m; ++i) {
          const float* xi = x + size_t(i) * size_t(k);
          for (int n = lo; n < hi; ++n)
            y[size_t(i) * size_t(w.n_out) + size_t(n)] = dot_f32(wbuf.data() + size_t(n - lo) * size_t(k), xi, k);
        }
        r0 += hi - lo;
      }
    }
  });
}

void matmul_bf16(ThreadPool& pool, const Bf16Weight& w, const float* x, int m, float* y) {
  const int k = w.k, block = 64;
  pool.parallel_for((w.n_out + block - 1) / block, 1, [&](int64_t b0, int64_t b1, int) {
    thread_local std::vector<float> wrow;
    wrow.resize(size_t(k));
    for (int64_t b = b0; b < b1; ++b) {
      int n0 = int(b) * block, n1 = std::min(w.n_out, n0 + block);
      for (int n = n0; n < n1; ++n) {
        const uint16_t* r = w.data + size_t(n) * size_t(k);
        for (int j = 0; j < k; ++j) wrow[size_t(j)] = bf16_to_f32(r[j]);
        for (int i = 0; i < m; ++i)
          y[size_t(i) * size_t(w.n_out) + size_t(n)] = dot_f32(wrow.data(), x + size_t(i) * size_t(k), k);
      }
    }
  });
}

void bf16_row(const uint16_t* table, int64_t row, int width, float* out) {
  const uint16_t* r = table + row * width;
  for (int j = 0; j < width; ++j) out[j] = bf16_to_f32(r[j]);
}

static float inv_norm(const float* x, int w) {
  float s = 0.f;
#pragma omp simd reduction(+ : s)
  for (int j = 0; j < w; ++j) s += x[j] * x[j];
  return 1.f / std::sqrt(s + 1e-6f);
}

void justnorm(float* x, int rows, int width) {
  for (int r = 0; r < rows; ++r) {
    float* v = x + size_t(r) * size_t(width);
    float f = inv_norm(v, width);
    for (int j = 0; j < width; ++j) v[j] *= f;
  }
}

void slerp_residual(float* a, const float* b, const float* lr, int rows, int width) {
  for (int r = 0; r < rows; ++r) {
    float* va = a + size_t(r) * size_t(width);
    const float* vb = b + size_t(r) * size_t(width);
    float fa = inv_norm(va, width), fb = inv_norm(vb, width);
    for (int j = 0; j < width; ++j) {
      float ya = va[j] * fa, yb = vb[j] * fb;
      va[j] = ya + lr[j] * (yb - ya);
    }
    float f = inv_norm(va, width);
    for (int j = 0; j < width; ++j) va[j] *= f;
  }
}

static const std::vector<float>& inv_freq(float theta, int d) {
  static std::mutex mutex;
  static std::map<std::pair<float, int>, std::vector<float>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto& f = cache[{theta, d}];
  if (f.empty()) {
    f.resize(size_t(d / 2));
    for (int i = 0; i < d / 2; ++i) f[size_t(i)] = 1.f / std::pow(theta, float(2 * i) / float(d));
  }
  return f;
}

void rope(float* x, int rows, int heads, int head_dim, int64_t pos0, float theta) {
  const std::vector<float>& f = inv_freq(theta, head_dim);
  const int half = head_dim / 2;
  std::vector<float> c(static_cast<size_t>(half)), s(static_cast<size_t>(half));
  for (int r = 0; r < rows; ++r) {
    float pos = float(pos0 + r);
    for (int i = 0; i < half; ++i) {
      float a = pos * f[size_t(i)];
      c[size_t(i)] = std::cos(a);
      s[size_t(i)] = std::sin(a);
    }
    for (int h = 0; h < heads; ++h) {
      float* v = x + (size_t(r) * size_t(heads) + size_t(h)) * size_t(head_dim);
      for (int i = 0; i < half; ++i) {
        float x1 = v[i], x2 = v[i + half];
        v[i] = x1 * c[size_t(i)] - x2 * s[size_t(i)];
        v[i + half] = x2 * c[size_t(i)] + x1 * s[size_t(i)];
      }
    }
  }
}

void swiglu_scaled(float* g, const float* u, const float* sg, const float* su, int rows, int w) {
  for (int r = 0; r < rows; ++r) {
    float* gr = g + size_t(r) * size_t(w);
    const float* ur = u + size_t(r) * size_t(w);
    for (int j = 0; j < w; ++j) {
      float z = gr[j] * sg[j];
      gr[j] = z / (1.f + std::exp(-z)) * (ur[j] * su[j]);
    }
  }
}

}  // namespace xtg
