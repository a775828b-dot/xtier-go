// NEON versions of the decode loops (aarch64 only). NVFP4: 16 codes per TBL lookup into twice the E2M1 values as
// int8, widened to FP32 and multiplied with the activations; the 0.5 comes back in the group scale. Same sums as
// the portable kernel up to FP32 rounding order (tests/kernels_test.cpp checks the difference).
#if defined(__aarch64__)
#include <arm_neon.h>

#include "xtg/fp.h"
#include "xtg/kernels.h"

namespace xtg {

bool neon_available() { return true; }   // NEON is part of every aarch64 CPU

void gemv_nvfp4_neon(const Nvfp4Weight& w, int row0, int row1, const float* x, float* y) {
  const int8x16_t lut = vld1q_s8(kE2m1x2);
  const uint8x8_t low = vdup_n_u8(15);
  const int groups = w.k / 16;
  for (int n = row0; n < row1; ++n) {
    const uint8_t* c = w.codes + size_t(n) * size_t(w.k / 2);
    const uint8_t* s = w.scales + size_t(n) * size_t(groups);
    float32x4_t acc0 = vdupq_n_f32(0.f), acc1 = vdupq_n_f32(0.f);
    for (int g = 0; g < groups; ++g) {
      uint8x8_t b = vld1_u8(c + g * 8);
      uint8x8x2_t z = vzip_u8(vand_u8(b, low), vshr_n_u8(b, 4));   // element 2j: low nibble of byte j
      int8x16_t v = vqtbl1q_s8(lut, vcombine_u8(z.val[0], z.val[1]));
      int16x8_t v0 = vmovl_s8(vget_low_s8(v)), v1 = vmovl_s8(vget_high_s8(v));
      const float* xg = x + g * 16;
      float32x4_t p = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v0))), vld1q_f32(xg));
      p = vfmaq_f32(p, vcvtq_f32_s32(vmovl_s16(vget_high_s16(v0))), vld1q_f32(xg + 4));
      p = vfmaq_f32(p, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v1))), vld1q_f32(xg + 8));
      p = vfmaq_f32(p, vcvtq_f32_s32(vmovl_s16(vget_high_s16(v1))), vld1q_f32(xg + 12));
      float sc = 0.5f * e4m3_to_f32(s[g]);
      if (g & 1)
        acc1 = vfmaq_n_f32(acc1, p, sc);
      else
        acc0 = vfmaq_n_f32(acc0, p, sc);
    }
    y[n] = vaddvq_f32(vaddq_f32(acc0, acc1)) * w.global;
  }
}

}  // namespace xtg
#endif
