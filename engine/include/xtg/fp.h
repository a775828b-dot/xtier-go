// Number formats: BF16, FP8 E4M3 (the NVFP4 block scales, the FP8 KV cache) and FP4 E2M1 (NVFP4 codes).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

namespace xtg {

inline float bf16_to_f32(uint16_t h) {
  uint32_t bits = uint32_t(h) << 16;
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

// Round to nearest even, as torch's float -> bfloat16 (NaN stays NaN).
inline uint16_t f32_to_bf16(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  if ((bits & 0x7fffffff) > 0x7f800000) return uint16_t((bits >> 16) | 0x40);
  bits += 0x7fff + ((bits >> 16) & 1);
  return uint16_t(bits >> 16);
}

// FP8 E4M3 (float8_e4m3fn: no infinities, 0x7f / 0xff are NaN, largest finite 448).
struct E4m3Table {
  float v[256];
};
extern const E4m3Table kE4m3;

inline float e4m3_to_f32(uint8_t b) { return kE4m3.v[b]; }

// FP32 -> E4M3 rounded to nearest even, saturating to +-448 (what the CUDA satfinite conversion gives).
uint8_t f32_to_e4m3(float f);

// E2M1 code (bit 3 sign, bits 0-2 magnitude) -> value: 0, 0.5, 1, 1.5, 2, 3, 4, 6.
constexpr float kE2m1[16] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f,
                             -0.f, -0.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};
// Twice the values, all integers: what the int8 kernels look up.
constexpr int8_t kE2m1x2[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

}  // namespace xtg
