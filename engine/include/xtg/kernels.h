// CPU kernels. Activations are FP32 everywhere; weights stay in their checkpoint format (NVFP4, BF16) and are
// decoded on the fly. The portable versions (kernels.cpp) are the reference and run on any CPU; aarch64 builds
// add NEON versions of the hot loops (kernels_neon.cpp), chosen at run time by the CPU's features.
#pragma once

#include <cstdint>
#include <vector>

#include "xtg/threadpool.h"

namespace xtg {

// An NVFP4 weight [n_out, k] in ModelOpt's layout: E2M1 codes two per byte (element 2j in the low nibble), one
// E4M3 scale per 16 inputs, and one FP32 scale for the tensor: w[n][j] = e2m1(code) * e4m3(scale[n][j/16]) * global.
struct Nvfp4Weight {
  const uint8_t* codes = nullptr;    // [n_out, k / 2]
  const uint8_t* scales = nullptr;   // [n_out, k / 16]
  float global = 1.f;
  int n_out = 0, k = 0;
};

struct Bf16Weight {
  const uint16_t* data = nullptr;    // [n_out, k]
  int n_out = 0, k = 0;
};

// y[m][n_out] = x[m][k] * W^T for several weights that read the same x, in one parallel region (q / k / v / gate
// of an attention block, gate / up of an MLP). ys[i] has room for m * ws[i].n_out values.
void matmul_nvfp4(ThreadPool& pool, const std::vector<const Nvfp4Weight*>& ws, const float* x, int m,
                  const std::vector<float*>& ys);
void matmul_bf16(ThreadPool& pool, const Bf16Weight& w, const float* x, int m, float* y);

// One row of a BF16 table (the embedding) to FP32.
void bf16_row(const uint16_t* table, int64_t row, int width, float* out);

// x / sqrt(sum(x^2) + 1e-6) over each of `rows` vectors of `width` (nGPT's justnorm), FP32 accumulation.
void justnorm(float* x, int rows, int width);
// out = justnorm(ya + lr * (yb - ya)) with ya = justnorm(a), yb = justnorm(b), per row (nGPT's slerp residual).
void slerp_residual(float* a, const float* b, const float* lr, int rows, int width);

// NeoX-style rotary embedding over [rows][heads][head_dim] at positions pos0 + row, as vLLM computes its table:
// inv_freq = 1 / theta^(2i / d) in FP32, angle = FP32(pos) * inv_freq in FP32.
void rope(float* x, int rows, int heads, int head_dim, int64_t pos0, float theta);

// silu(g * sg) * (u * su), elementwise over rows of width w (sg, su: per-column scales); written to g.
void swiglu_scaled(float* g, const float* u, const float* sg, const float* su, int rows, int w);

// Debug / test helpers: the decoded weight row.
void nvfp4_row(const Nvfp4Weight& w, int n, float* out);

enum class KernelSet { Portable, Neon };
KernelSet kernel_set();
const char* kernel_set_name();
void set_kernel_set(KernelSet k);   // tests compare the two

}  // namespace xtg
