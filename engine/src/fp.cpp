#include "xtg/fp.h"

namespace xtg {

static constexpr float pow2i(int e) {
  float r = 1.f;
  for (int i = 0; i < e; ++i) r *= 2.f;
  for (int i = 0; i < -e; ++i) r *= 0.5f;
  return r;
}

static constexpr E4m3Table make_e4m3() {
  E4m3Table t{};
  for (int b = 0; b < 256; ++b) {
    int e = (b >> 3) & 15, m = b & 7;
    float mag = 0.f;
    if ((b & 0x7f) == 0x7f)
      mag = __builtin_nanf("");
    else if (e == 0)
      mag = float(m) * pow2i(-9);   // subnormal
    else
      mag = (1.f + float(m) / 8.f) * pow2i(e - 7);
    t.v[b] = (b & 0x80) ? -mag : mag;
  }
  return t;
}

constexpr E4m3Table kE4m3 = make_e4m3();

uint8_t f32_to_e4m3(float f) {
  uint8_t sign = std::signbit(f) ? 0x80 : 0;
  if (std::isnan(f)) return uint8_t(sign | 0x7f);
  float a = std::fabs(f);
  if (a >= 448.f) return uint8_t(sign | 0x7e);
  // Codes 0x00..0x7e grow with the magnitude: the largest code not above a, then round to nearest even.
  int lo = 0, hi = 0x7e;
  while (lo < hi) {
    int mid = (lo + hi + 1) / 2;
    if (kE4m3.v[mid] <= a)
      lo = mid;
    else
      hi = mid - 1;
  }
  if (lo < 0x7e) {
    float d0 = a - kE4m3.v[lo], d1 = kE4m3.v[lo + 1] - a;
    if (d1 < d0 || (d1 == d0 && (lo & 1))) ++lo;
  }
  return uint8_t(sign | lo);
}

}  // namespace xtg
