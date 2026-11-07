#include "xtg/opencl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "xtg/common.h"
#include "xtg/fp.h"
using namespace xtg;
int main() {
  try {
    std::mt19937 rng(42);
    auto rnd = [&] { return float(int(rng() % 2001) - 1000) / 1000; };
    for (auto shape : {std::pair{70, 256}, std::pair{7168, 2560}, std::pair{2560, 7168}}) {
      OpenClBackend gpu;
      int n = shape.first, k = shape.second;
      std::vector<uint8_t> codes(size_t(n) * k / 2), scales(size_t(n) * k / 16);
      for (auto& v : codes) v = rng() & 255;
      for (auto& v : scales) v = 0x20 + rng() % 48;
      Nvfp4Weight w{codes.data(), scales.data(), .37f, n, k};
      gpu.prepare(w);
      {
        std::vector<float> x(k), a(n), b(n), c(n), d(n);
        for (auto& v : x) v = rnd();
        gpu.matmul({&w, &w, &w, &w}, x.data(), 1, {a.data(), b.data(), c.data(), d.data()});
        XTG_CHECK(a == b && a == c && a == d, "Fused decode projection offsets differ");
        gpu.matmul({&w}, x.data(), 1, {b.data()});
        XTG_CHECK(a == b, "Fused decode differs from single projection");
      }
      for (int m : {1, 2, 7, 65, 512}) {
        std::vector<float> x(size_t(m) * k), y(size_t(m) * n), row(k);
        for (auto& v : x) v = rnd();
        gpu.matmul({&w}, x.data(), m, {y.data()});
        double times[3];
        for (double& t : times) {
          double st = now_s();
          gpu.matmul({&w}, x.data(), m, {y.data()});
          t = now_s() - st;
        }
        std::sort(times, times + 3);
        double elapsed = times[1];
        double err = 0, sq = 0;
        int count = 0;
        for (int r : {0, m - 1})
          for (int j = 0; j < 32; j++) {
            int col = (j * 137 + 3) % n;
            nvfp4_row(w, col, row.data());
            double ref = 0;
            for (int z = 0; z < k; z++) ref += double(row[z]) * x[size_t(r) * k + z];
            sq += ref * ref;
            err = std::max(err, std::abs(ref - y[size_t(r) * n + col]));
            count++;
          }
        double norm = err / std::sqrt(sq / count);
        XTG_CHECK(norm < 3e-5, "NVFP4 error %g", norm);
        printf("NVFP4 M=%d N=%d K=%d %.3f ms %.2f GFLOP/s rms_error=%g\n", m, n, k, elapsed * 1000,
               2.0 * m * n * k / elapsed / 1e9, norm);
        fflush(stdout);
      }
    }
    const int n = 131072, k = 2560;
    OpenClBackend gpu;
    std::vector<uint16_t> w(size_t(n) * k);
    for (auto& v : w) v = f32_to_bf16(rnd());
    Bf16Weight b{w.data(), n, k};
    gpu.prepare(b);
    for (int m : {1, 2, 16}) {
      std::vector<float> x(size_t(m) * k), y(size_t(m) * n);
      for (auto& v : x) v = rnd();
      gpu.matmul(b, x.data(), m, y.data());
      double times[3];
      for (double& t : times) {
        double st = now_s();
        gpu.matmul(b, x.data(), m, y.data());
        t = now_s() - st;
      }
      std::sort(times, times + 3);
      double elapsed = times[1];
      double err = 0, sq = 0;
      int count = 0;
      for (int r : {0, m - 1})
        for (int j = 0; j < 32; j++) {
          int col = (j * 4093 + 3) % n;
          double ref = 0;
          for (int z = 0; z < k; z++) ref += double(bf16_to_f32(w[size_t(col) * k + z])) * x[size_t(r) * k + z];
          sq += ref * ref;
          err = std::max(err, std::abs(ref - y[size_t(r) * n + col]));
          count++;
        }
      double norm = err / std::sqrt(sq / count);
      XTG_CHECK(norm < 3e-5, "BF16 error %g", norm);
      printf("BF16 M=%d N=%d K=%d %.3f ms %.2f GFLOP/s rms_error=%g\n", m, n, k, elapsed * 1000,
             2.0 * m * n * k / elapsed / 1e9, norm);
      fflush(stdout);
    }
    puts("PASS OpenCL real-shape kernels");
    return 0;
  } catch (const std::exception& e) {
    fprintf(stderr, "FAIL %s\n", e.what());
    return 1;
  }
}
