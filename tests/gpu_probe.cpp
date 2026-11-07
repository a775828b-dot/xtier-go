// gpu_probe: the building blocks of a GPU-resident Dory step, measured on the device (adb shell; also runs on any
// OpenCL GPU for a correctness check). Prints one line per measurement:
//   device / extensions          what the driver offers (recordable queues, integer dot product, subgroups, fp16)
//   launch async|sync|roundtrip  microseconds per kernel: queued back to back, clFinish after each, and the
//                                current backend's pattern (upload 10 KiB, kernel, read 28 KiB, clFinish)
//   read device|hostptr          streaming read bandwidth of a 512 MiB buffer, driver-owned and ALLOC_HOST_PTR
//   gemv4 M=1..16                NVFP4 7168 x 2560 against M activation rows at once (what speculative decoding
//                                needs): time per call and the weight bytes per second
//   gemvt                        the same with the weights k-major (one row per work-item) on Dory's projection
//                                shapes, global loads or an image1d_buffer, K split over 1..8 work-items, M = 1..16
//   peak fp32|fp16|int8dot       register-only multiply-add throughput
// Every kernel result is checked against the CPU.
#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "xtg/fp.h"

namespace {
#define API(X)                                                                                              \
  X(clGetPlatformIDs) X(clGetDeviceIDs) X(clGetDeviceInfo) X(clCreateContext) X(clCreateCommandQueue)      \
  X(clCreateProgramWithSource) X(clBuildProgram) X(clGetProgramBuildInfo) X(clCreateKernel) X(clCreateBuffer) \
  X(clSetKernelArg) X(clEnqueueNDRangeKernel) X(clEnqueueWriteBuffer) X(clEnqueueReadBuffer) X(clFinish)    \
  X(clReleaseMemObject) X(clEnqueueMapBuffer) X(clEnqueueUnmapMemObject) X(clGetEventProfilingInfo)          \
  X(clWaitForEvents) X(clReleaseEvent) X(clCreateImage)
#define DECL(n) decltype(&::n) p_##n = nullptr;
API(DECL)
#undef DECL

void die(const char* what, cl_int e) {
  std::fprintf(stderr, "FAIL %s: %d\n", what, e);
  std::exit(1);
}
#define CK(call)                       \
  do {                                 \
    cl_int e_ = (call);                \
    if (e_ != CL_SUCCESS) die(#call, e_); \
  } while (0)

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

const char* kSource = R"CLC(
#ifndef MR
#define MR 1
#endif
__kernel void tiny(__global float* y) { int i = get_global_id(0); y[i] = y[i] * 0.5f + 1.0f; }

// sum of a big buffer in uint4 loads; one partial per work-group
__kernel void readsum(__global const uint4* p, ulong n, __global uint* out) {
  uint acc = 0;
  for (ulong i = get_global_id(0); i < n; i += get_global_size(0)) { uint4 v = p[i]; acc += v.x ^ v.y ^ v.z ^ v.w; }
  __local uint s[256];
  int l = get_local_id(0); s[l] = acc; barrier(CLK_LOCAL_MEM_FENCE);
  for (int k = 128; k; k >>= 1) { if (l < k) s[l] += s[l + k]; barrier(CLK_LOCAL_MEM_FENCE); }
  if (l == 0) out[get_group_id(0)] = s[0];
}

float e4m3(uint b) {
  uint e = (b >> 3) & 15, m = b & 7;
  float v = e ? as_float(((e + 120) << 23) | (m << 20)) : (float)m * 0.001953125f;
  return (b & 128) ? -v : v;
}
float4 e2m1x4(uint c) {   // four codes in the low 16 bits, low nibble first
  uint4 code = (uint4)(c & 15, (c >> 4) & 15, (c >> 8) & 15, (c >> 12) & 15), mag = code & 7, ex = mag >> 1;
  float4 v = as_float4(((ex + 126) << 23) | ((mag & 1) << 22));
  v = select(v, convert_float4(mag) * .5f, mag < 2);
  return as_float4(as_uint4(v) | ((code & 8) << 28));
}
// NVFP4 rows against MR activation rows: 32 lanes per output row, 4 rows per work-group, FP32 accumulation;
// every 16 weights are decoded once and used for all MR rows
__kernel void gemv4m(__global const uchar* c, __global const uchar* s, __global const float* x, __global float* y,
                     int K, int N, float scale) {
  int lid = get_local_id(0), lane = lid & 31, row = get_group_id(0) * 4 + lid / 32;
  float acc[MR];
  for (int m = 0; m < MR; m++) acc[m] = 0;
  if (row < N)
    for (int g = lane; g < K / 16; g += 32) {
      uint4 q = vload4(0, (__global const uint*)(c + (size_t)row * (K / 2) + g * 8)).xyzy;
      float sc = e4m3(s[(size_t)row * (K / 16) + g]);
      float4 w0 = e2m1x4(q.x) * sc, w1 = e2m1x4(q.x >> 16) * sc, w2 = e2m1x4(q.y) * sc, w3 = e2m1x4(q.y >> 16) * sc;
      for (int m = 0; m < MR; m++) {
        __global const float* xm = x + (size_t)m * K + g * 16;
        float4 t = w0 * vload4(0, xm);
        t = mad(w1, vload4(1, xm), t); t = mad(w2, vload4(2, xm), t); t = mad(w3, vload4(3, xm), t);
        acc[m] += t.x + t.y + t.z + t.w;
      }
    }
  __local float red[128];
  for (int m = 0; m < MR; m++) {
    red[lid] = acc[m]; barrier(CLK_LOCAL_MEM_FENCE);
    for (int k = 16; k; k >>= 1) { if (lane < k) red[lid] += red[lid + k]; barrier(CLK_LOCAL_MEM_FENCE); }
    if (lane == 0 && row < N) y[(size_t)m * N + row] = red[lid] * scale;
    barrier(CLK_LOCAL_MEM_FENCE);
  }
}

// The same product with the weights k-major: codes [K/32][N] as uint4 (the 32 codes of one row, low nibble first),
// scales [K/32][N] as ushort (two E4M3, the low byte for the first 16). Rows go to consecutive work-items, so a wave
// reads consecutive 16-byte blocks and the activations are the same address for all of it; the KS work-items of a
// row (local size L0 x KS, L0 x KS <= 256) split K and add up in local memory. Variants: IMG reads the codes through
// an image1d_buffer, XCONST the activations from __constant memory, ROWS > 1 rows per work-item (r, r + G, ...)
// with the activations loaded once for all of them.
#ifndef KS
#define KS 1
#endif
#ifndef ROWS
#define ROWS 1
#endif
#ifdef IMG
#define CODES __read_only image1d_buffer_t c
#define LOADQ(i) read_imageui(c, (int)(i))
#else
#define CODES __global const uint4* c
#define LOADQ(i) c[i]
#endif
#ifdef XCONST
#define XPTR __constant const float*
#else
#define XPTR __global const float*
#endif
float dot32(uint4 q, uint sc, float4 x0, float4 x1, float4 x2, float4 x3, float4 x4, float4 x5, float4 x6, float4 x7) {
  float4 g0 = e2m1x4(q.x) * x0, g1 = e2m1x4(q.z) * x4;
  g0 = mad(e2m1x4(q.x >> 16), x1, g0); g0 = mad(e2m1x4(q.y), x2, g0); g0 = mad(e2m1x4(q.y >> 16), x3, g0);
  g1 = mad(e2m1x4(q.z >> 16), x5, g1); g1 = mad(e2m1x4(q.w), x6, g1); g1 = mad(e2m1x4(q.w >> 16), x7, g1);
  return mad(e4m3(sc & 255), g0.x + g0.y + g0.z + g0.w, e4m3(sc >> 8) * (g1.x + g1.y + g1.z + g1.w));
}
__kernel void gemvt(CODES, __global const ushort* s, XPTR x, __global float* y, int K, int N, float scale) {
  int g = get_global_id(0), G = get_global_size(0), lr = get_local_id(0), ks = get_local_id(1), L0 = get_local_size(0);
  int nb = K / 32, per = (nb + KS - 1) / KS, b0 = ks * per, b1 = min(nb, b0 + per);
  float acc[MR][ROWS];
#pragma unroll
  for (int m = 0; m < MR; m++)
#pragma unroll
    for (int j = 0; j < ROWS; j++) acc[m][j] = 0;
  for (int b = b0; b < b1; b++) {
    uint4 q[ROWS];
    uint sc[ROWS];
#pragma unroll
    for (int j = 0; j < ROWS; j++) {
      int r = min(g + j * G, N - 1);
      q[j] = LOADQ((size_t)b * N + r);
      sc[j] = s[(size_t)b * N + r];
    }
#pragma unroll
    for (int m = 0; m < MR; m++) {
      XPTR xm = x + (size_t)m * K + b * 32;
      float4 x0 = vload4(0, xm), x1 = vload4(1, xm), x2 = vload4(2, xm), x3 = vload4(3, xm);
      float4 x4 = vload4(4, xm), x5 = vload4(5, xm), x6 = vload4(6, xm), x7 = vload4(7, xm);
#pragma unroll
      for (int j = 0; j < ROWS; j++) acc[m][j] += dot32(q[j], sc[j], x0, x1, x2, x3, x4, x5, x6, x7);
    }
  }
  __local float red[KS][256 / KS];
#pragma unroll
  for (int m = 0; m < MR; m++)
#pragma unroll
    for (int j = 0; j < ROWS; j++) {
      int r = g + j * G;
      if (KS > 1) {
        red[ks][lr] = acc[m][j];
        barrier(CLK_LOCAL_MEM_FENCE);
        if (ks == 0) for (int i = 1; i < KS; i++) acc[m][j] += red[i][lr];
        barrier(CLK_LOCAL_MEM_FENCE);
      }
      if (ks == 0 && r < N) y[(size_t)m * N + r] = acc[m][j] * scale;
    }
}

// Prefill GEMM prototype: C[M][N] = A[M][K] . W[N][K]^T with NVFP4 W decoded once per tile into local memory.
// 256 work-items, each 4 rows of A x 8 columns of W (a 64 x 128 tile), K in steps of 32 with a T-precision
// partial per step added into FP32 (T = half where the device has cl_khr_fp16, else float for checking).
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
typedef half T; typedef half4 T4; typedef half8 T8;
#define CVT8 convert_half8
#else
typedef float T; typedef float4 T4; typedef float8 T8;
#define CVT8 convert_float8
#endif
__kernel void gemm_t(__global const T* A, __global const uchar* c, __global const uchar* s, __global float* C,
                     int M, int N, int K, float scale) {
  __local T As[64][32 + 2];
  __local T Bs[32][128];
  const int tid = get_local_id(0), tm = tid / 16, tn = tid % 16;
  const int m0 = get_group_id(1) * 64, n0 = get_group_id(0) * 128;
  float8 acc0 = 0, acc1 = 0, acc2 = 0, acc3 = 0;
  for (int k0 = 0; k0 < K; k0 += 32) {
    {  // A: 64 x 32, eight values per work-item
      int r = tid / 4, kk = (tid % 4) * 8;
      vstore8(m0 + r < M ? vload8(0, A + (size_t)(m0 + r) * K + k0 + kk) : (T8)0, 0, &As[r][kk]);
    }
    {  // W: 128 rows x 32 k, sixteen weights (one NVFP4 block) per work-item, decoded into k-major Bs
      int r = tid / 2, half16 = (tid % 2) * 16, n = n0 + r;
      if (n < N) {
        uint2 q = vload2(0, (__global const uint*)(c + (size_t)n * (K / 2) + (k0 + half16) / 2));
        float sc = e4m3(s[(size_t)n * (K / 16) + (k0 + half16) / 16]);
        float4 w0 = e2m1x4(q.x) * sc, w1 = e2m1x4(q.x >> 16) * sc, w2 = e2m1x4(q.y) * sc, w3 = e2m1x4(q.y >> 16) * sc;
        float wv[16] = {w0.x, w0.y, w0.z, w0.w, w1.x, w1.y, w1.z, w1.w, w2.x, w2.y, w2.z, w2.w, w3.x, w3.y, w3.z, w3.w};
        for (int i = 0; i < 16; i++) Bs[half16 + i][r] = (T)wv[i];
      } else for (int i = 0; i < 16; i++) Bs[half16 + i][r] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    T8 p0 = 0, p1 = 0, p2 = 0, p3 = 0;
    for (int k = 0; k < 32; k++) {
      T8 b = vload8(0, &Bs[k][tn * 8]);
      p0 = mad((T8)As[tm * 4 + 0][k], b, p0); p1 = mad((T8)As[tm * 4 + 1][k], b, p1);
      p2 = mad((T8)As[tm * 4 + 2][k], b, p2); p3 = mad((T8)As[tm * 4 + 3][k], b, p3);
    }
    acc0 += convert_float8(p0); acc1 += convert_float8(p1); acc2 += convert_float8(p2); acc3 += convert_float8(p3);
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  float8 acc[4] = {acc0, acc1, acc2, acc3};
  for (int i = 0; i < 4; i++) {
    int m = m0 + tm * 4 + i, n = n0 + tn * 8;
    if (m < M && n + 7 < N) vstore8(acc[i] * scale, 0, C + (size_t)m * N + n);
  }
}

__kernel void peak_fp32(__global float* out, float a, float b) {
  float4 x0 = (float4)(get_global_id(0)), x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3;
  for (int i = 0; i < 1024; i++) { x0 = mad(x0, a, b); x1 = mad(x1, a, b); x2 = mad(x2, a, b); x3 = mad(x3, a, b); }
  out[get_global_id(0)] = dot(x0 + x1 + x2 + x3, (float4)1);
}
#ifdef cl_khr_fp16
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
__kernel void peak_fp16(__global float* out, float fa, float fb) {
  half a = (half)fa, b = (half)fb;
  half8 x0 = (half8)((half)get_local_id(0)), x1 = x0 + (half)1, x2 = x0 + (half)2, x3 = x0 + (half)3;
  for (int i = 0; i < 1024; i++) { x0 = mad(x0, a, b); x1 = mad(x1, a, b); x2 = mad(x2, a, b); x3 = mad(x3, a, b); }
  half8 s = x0 + x1 + x2 + x3;
  out[get_global_id(0)] = (float)(s.s0 + s.s1 + s.s2 + s.s3 + s.s4 + s.s5 + s.s6 + s.s7);
}
#endif
#ifdef cl_khr_integer_dot_product
__kernel void peak_int8dot(__global int* out, uint a, uint b) {
  int x0 = get_global_id(0), x1 = x0 + 1, x2 = x0 + 2, x3 = x0 + 3, x4 = x0 + 4, x5 = x0 + 5, x6 = x0 + 6, x7 = x0 + 7;
  for (int i = 0; i < 1024; i++) {
    x0 = dot_acc_sat_4x8packed_ss_int(a, b, x0); x1 = dot_acc_sat_4x8packed_ss_int(a, b, x1);
    x2 = dot_acc_sat_4x8packed_ss_int(a, b, x2); x3 = dot_acc_sat_4x8packed_ss_int(a, b, x3);
    x4 = dot_acc_sat_4x8packed_ss_int(a, b, x4); x5 = dot_acc_sat_4x8packed_ss_int(a, b, x5);
    x6 = dot_acc_sat_4x8packed_ss_int(a, b, x6); x7 = dot_acc_sat_4x8packed_ss_int(a, b, x7);
  }
  out[get_global_id(0)] = x0 + x1 + x2 + x3 + x4 + x5 + x6 + x7;
}
#endif
)CLC";

struct Cl {
  cl_device_id dev{};
  cl_context ctx{};
  cl_command_queue q{};
  std::string ext;

  void init() {
    void* lib = dlopen("libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("/vendor/lib64/libOpenCL.so", RTLD_NOW | RTLD_LOCAL);
    if (!lib) lib = dlopen("libOpenCL.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) { std::fprintf(stderr, "FAIL no OpenCL driver: %s\n", dlerror()); std::exit(1); }
#define LOAD(n) p_##n = reinterpret_cast<decltype(p_##n)>(dlsym(lib, #n)); if (!p_##n) die("dlsym " #n, -1);
    API(LOAD)
#undef LOAD
    cl_platform_id ps[8];
    cl_uint np = 0;
    CK(p_clGetPlatformIDs(8, ps, &np));
    for (cl_uint i = 0; i < np && !dev; i++) p_clGetDeviceIDs(ps[i], CL_DEVICE_TYPE_GPU, 1, &dev, nullptr);
    if (!dev) die("no GPU device", -1);
    cl_int e;
    ctx = p_clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
    CK(e);
    q = p_clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &e);
    CK(e);
    ext = str(CL_DEVICE_EXTENSIONS);
  }
  std::string str(cl_device_info what) {
    size_t n = 0;
    p_clGetDeviceInfo(dev, what, 0, nullptr, &n);
    std::string s(n, '\0');
    p_clGetDeviceInfo(dev, what, n, s.data(), nullptr);
    while (!s.empty() && s.back() == '\0') s.pop_back();
    return s;
  }
  template <class T> T num(cl_device_info what) {
    T v{};
    p_clGetDeviceInfo(dev, what, sizeof(v), &v, nullptr);
    return v;
  }
  bool has(const char* e) const { return ext.find(e) != std::string::npos; }
  cl_kernel kernel(const std::string& opts, const char* name) {
    cl_int e;
    cl_program p = p_clCreateProgramWithSource(ctx, 1, &kSource, nullptr, &e);
    CK(e);
    if (p_clBuildProgram(p, 1, &dev, opts.c_str(), nullptr, nullptr) != CL_SUCCESS) {
      char log[16384] = {};
      p_clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, sizeof(log), log, nullptr);
      std::fprintf(stderr, "FAIL build (%s): %s\n", opts.c_str(), log);
      std::exit(1);
    }
    cl_kernel k = p_clCreateKernel(p, name, &e);
    CK(e);
    return k;
  }
  cl_mem buffer(cl_mem_flags f, size_t bytes, void* host = nullptr) {
    cl_int e;
    cl_mem b = p_clCreateBuffer(ctx, f, bytes, host, &e);
    CK(e);
    return b;
  }
  template <class T> void arg(cl_kernel k, int i, const T& v) { CK(p_clSetKernelArg(k, i, sizeof(T), &v)); }
  // kernel time from the profiling event (seconds)
  double run(cl_kernel k, size_t global, size_t local) {
    cl_event ev;
    CK(p_clEnqueueNDRangeKernel(q, k, 1, nullptr, &global, local ? &local : nullptr, 0, nullptr, &ev));
    return wait(ev);
  }
  double run2(cl_kernel k, size_t g0, size_t g1, size_t l0, size_t l1) {
    size_t g[2] = {g0, g1}, l[2] = {l0, l1};
    cl_event ev;
    CK(p_clEnqueueNDRangeKernel(q, k, 2, nullptr, g, l, 0, nullptr, &ev));
    return wait(ev);
  }
  double wait(cl_event ev) {
    CK(p_clWaitForEvents(1, &ev));
    cl_ulong a = 0, b = 0;
    p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(a), &a, nullptr);
    p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(b), &b, nullptr);
    p_clReleaseEvent(ev);
    return double(b - a) * 1e-9;
  }
};

double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

void launch(Cl& c) {
  cl_kernel k = c.kernel("", "tiny");
  cl_mem y = c.buffer(CL_MEM_READ_WRITE, 1 << 20);
  std::vector<float> host(7168 + 2560);
  cl_mem in = c.buffer(CL_MEM_READ_WRITE, 2560 * 4), out = c.buffer(CL_MEM_READ_WRITE, 7168 * 4);
  c.arg(k, 0, y);
  size_t g = 256, l = 64;
  for (int i = 0; i < 50; i++) CK(p_clEnqueueNDRangeKernel(c.q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr));
  CK(p_clFinish(c.q));
  const int n = 2000;
  double t = now();
  for (int i = 0; i < n; i++) CK(p_clEnqueueNDRangeKernel(c.q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr));
  CK(p_clFinish(c.q));
  std::printf("launch async %.1f us per kernel (%d queued, one clFinish)\n", (now() - t) / n * 1e6, n);
  const int m = 300;
  t = now();
  for (int i = 0; i < m; i++) {
    CK(p_clEnqueueNDRangeKernel(c.q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr));
    CK(p_clFinish(c.q));
  }
  std::printf("launch sync %.1f us per kernel (clFinish after each)\n", (now() - t) / m * 1e6);
  t = now();
  for (int i = 0; i < m; i++) {
    CK(p_clEnqueueWriteBuffer(c.q, in, CL_FALSE, 0, 2560 * 4, host.data(), 0, nullptr, nullptr));
    CK(p_clEnqueueNDRangeKernel(c.q, k, 1, nullptr, &g, &l, 0, nullptr, nullptr));
    CK(p_clEnqueueReadBuffer(c.q, out, CL_FALSE, 0, 7168 * 4, host.data(), 0, nullptr, nullptr));
    CK(p_clFinish(c.q));
  }
  std::printf("launch roundtrip %.1f us per call (upload 10 KiB, kernel, read 28 KiB, clFinish: today's backend)\n",
              (now() - t) / m * 1e6);
  p_clReleaseMemObject(y);
  p_clReleaseMemObject(in);
  p_clReleaseMemObject(out);
}

void bandwidth(Cl& c) {
  cl_kernel k = c.kernel("", "readsum");
  const size_t bytes = size_t(512) << 20, n4 = bytes / 16;
  const size_t groups = 4096, local = 256;
  cl_mem out = c.buffer(CL_MEM_READ_WRITE, groups * 4);
  for (int hostptr = 0; hostptr < 2; hostptr++) {
    cl_mem b = c.buffer(hostptr ? (CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR) : CL_MEM_READ_ONLY, bytes);
    // fill through a map so both kinds hold the same data and the pages exist
    cl_int e;
    void* p = p_clEnqueueMapBuffer(c.q, b, CL_TRUE, CL_MAP_WRITE, 0, bytes, 0, nullptr, nullptr, &e);
    CK(e);
    std::memset(p, 0x5a, bytes);
    CK(p_clEnqueueUnmapMemObject(c.q, b, p, 0, nullptr, nullptr));
    CK(p_clFinish(c.q));
    cl_ulong nn = n4;
    c.arg(k, 0, b);
    c.arg(k, 1, nn);
    c.arg(k, 2, out);
    std::vector<double> ts;
    for (int i = 0; i < 6; i++) ts.push_back(c.run(k, groups * local, local));
    double t = median(ts);
    std::printf("read %s %.1f GB/s (512 MiB, kernel time %.2f ms)\n", hostptr ? "hostptr" : "device", bytes / t / 1e9,
                t * 1e3);
    p_clReleaseMemObject(b);
  }
  p_clReleaseMemObject(out);
}

void gemv(Cl& c) {
  const int N = 7168, K = 2560;
  std::mt19937 rng(7);
  std::vector<uint8_t> codes(size_t(N) * K / 2), scales(size_t(N) * K / 16);
  for (auto& v : codes) v = uint8_t(rng());
  for (auto& v : scales) v = uint8_t(0x30 + rng() % 16);   // E4M3 around 0.5 .. 1
  const float global = 0.01f;
  cl_mem cb = c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, codes.size(), codes.data());
  cl_mem sb = c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scales.size(), scales.data());
  const double bytes = double(codes.size() + scales.size());
  double t1 = 0;
  for (int M : {1, 2, 4, 8, 16}) {
    cl_kernel k = c.kernel("-DMR=" + std::to_string(M), "gemv4m");
    std::vector<float> x(size_t(M) * K), y(size_t(M) * N);
    std::normal_distribution<float> nd(0.f, 0.02f);
    for (auto& v : x) v = nd(rng);
    cl_mem xb = c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.size() * 4, x.data());
    cl_mem yb = c.buffer(CL_MEM_WRITE_ONLY, y.size() * 4);
    c.arg(k, 0, cb); c.arg(k, 1, sb); c.arg(k, 2, xb); c.arg(k, 3, yb);
    c.arg(k, 4, K); c.arg(k, 5, N); c.arg(k, 6, global);
    std::vector<double> ts;
    for (int i = 0; i < 9; i++) ts.push_back(c.run(k, size_t((N + 3) / 4) * 128, 128));
    double t = median(ts);
    CK(p_clEnqueueReadBuffer(c.q, yb, CL_TRUE, 0, y.size() * 4, y.data(), 0, nullptr, nullptr));
    // check two rows of every activation vector against the CPU
    double err = 0, ref2 = 0;
    for (int m = 0; m < M; m++)
      for (int n : {0, N - 1}) {
        double r = 0;
        for (int kk = 0; kk < K; kk++) {
          uint8_t b = codes[size_t(n) * K / 2 + kk / 2];
          int code = (kk & 1) ? b >> 4 : b & 15;
          r += double(xtg::kE2m1[code]) * xtg::e4m3_to_f32(scales[size_t(n) * K / 16 + kk / 16]) * global * x[size_t(m) * K + kk];
        }
        err = std::max(err, std::fabs(r - y[size_t(m) * N + n]));
        ref2 += r * r;
      }
    double rms = std::sqrt(ref2 / (2 * M));
    if (M == 1) t1 = t;
    std::printf("gemv4 M=%d %.3f ms per call (%.2fx the M=1 time), %.1f GB/s of weights, err/rms %.1e%s\n", M, t * 1e3,
                t / t1, bytes / t / 1e9, err / rms, err / rms < 1e-4 ? "" : "  MISMATCH");
    p_clReleaseMemObject(xb);
    p_clReleaseMemObject(yb);
  }
  p_clReleaseMemObject(cb);
  p_clReleaseMemObject(sb);
}

// The k-major GEMV (gemvt) against gemv4m on Dory's four projection shapes: codes through global loads or an
// image1d_buffer, KS work-items per row; then M = 2..16 activation rows for the fastest variant of each shape
void gemvt(Cl& c) {
  struct Shape { const char* name; int N, K; };
  std::mt19937 rng(11);
  const size_t img_max = c.num<size_t>(CL_DEVICE_IMAGE_MAX_BUFFER_SIZE);
  const int MMAX = 16;
  for (Shape sh : {Shape{"qkvg", 10240, 2560}, Shape{"o", 2560, 4096}, Shape{"gu", 14336, 2560}, Shape{"down", 2560, 7168}}) {
    const int N = sh.N, K = sh.K, nb = K / 32;
    std::vector<uint8_t> codes(size_t(N) * K / 2), scales(size_t(N) * K / 16), ct(codes.size());
    std::vector<uint16_t> st(size_t(nb) * N);
    for (auto& v : codes) v = uint8_t(rng());
    for (auto& v : scales) v = uint8_t(0x30 + rng() % 16);
    for (int b = 0; b < nb; b++)
      for (int r = 0; r < N; r++) {
        std::memcpy(&ct[(size_t(b) * N + r) * 16], &codes[size_t(r) * K / 2 + size_t(b) * 16], 16);
        st[size_t(b) * N + r] = uint16_t(scales[size_t(r) * K / 16 + 2 * b] | (scales[size_t(r) * K / 16 + 2 * b + 1] << 8));
      }
    std::vector<float> x(size_t(MMAX) * K), y(size_t(MMAX) * N);
    std::normal_distribution<float> nd(0.f, 0.02f);
    for (auto& v : x) v = nd(rng);
    const float global = 0.01f;
    const double bytes = double(codes.size() + scales.size());
    cl_mem cb = c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, codes.size(), codes.data());
    cl_mem sb = c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scales.size(), scales.data());
    cl_mem ctb = c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ct.size(), ct.data());
    cl_mem stb = c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, st.size() * 2, st.data());
    cl_mem xb = c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, x.size() * 4, x.data());
    cl_mem yb = c.buffer(CL_MEM_WRITE_ONLY, y.size() * 4);
    cl_mem img = nullptr;
    if (size_t(N) * nb <= img_max) {
      cl_image_format fmt{CL_RGBA, CL_UNSIGNED_INT32};
      cl_image_desc desc{};
      desc.image_type = CL_MEM_OBJECT_IMAGE1D_BUFFER;
      desc.image_width = size_t(N) * nb;
      desc.buffer = ctb;
      cl_int e;
      img = p_clCreateImage(c.ctx, CL_MEM_READ_ONLY, &fmt, &desc, nullptr, &e);
      if (e != CL_SUCCESS) img = nullptr;
    }
    auto error = [&](int M) {   // rows 0 and N - 1 of every activation vector against the CPU
      CK(p_clEnqueueReadBuffer(c.q, yb, CL_TRUE, 0, size_t(M) * N * 4, y.data(), 0, nullptr, nullptr));
      double err = 0, ref2 = 0;
      for (int m = 0; m < M; m++)
        for (int n : {0, N - 1}) {
          double r = 0;
          for (int kk = 0; kk < K; kk++) {
            uint8_t b = codes[size_t(n) * K / 2 + kk / 2];
            int code = (kk & 1) ? b >> 4 : b & 15;
            r += double(xtg::kE2m1[code]) * xtg::e4m3_to_f32(scales[size_t(n) * K / 16 + kk / 16]) * global * x[size_t(m) * K + kk];
          }
          err = std::max(err, std::fabs(r - y[size_t(m) * N + n]));
          ref2 += r * r;
        }
      return err / std::sqrt(ref2 / (2 * M));
    };
    cl_kernel base = c.kernel("-DMR=1", "gemv4m");
    c.arg(base, 0, cb); c.arg(base, 1, sb); c.arg(base, 2, xb); c.arg(base, 3, yb);
    c.arg(base, 4, K); c.arg(base, 5, N); c.arg(base, 6, global);
    std::vector<double> ts;
    for (int i = 0; i < 9; i++) ts.push_back(c.run(base, size_t((N + 3) / 4) * 128, 128));
    const double tb = median(ts);
    std::printf("gemvt %s N=%d K=%d: gemv4m %.3f ms, %.1f GB/s, err/rms %.1e\n", sh.name, N, K, tb * 1e3, bytes / tb / 1e9,
                error(1));
    struct V { bool image, xconst; int rows, ks; };
    auto label = [](V v) {
      return std::string(v.image ? "image" : "global") + (v.xconst ? "+xconst" : "") + " rows=" + std::to_string(v.rows) +
             " KS=" + std::to_string(v.ks);
    };
    auto time = [&](V v, int M) {
      std::string opts = "-DMR=" + std::to_string(M) + " -DKS=" + std::to_string(v.ks) + " -DROWS=" + std::to_string(v.rows) +
                         (v.image ? " -DIMG" : "") + (v.xconst ? " -DXCONST" : "");
      cl_kernel k = c.kernel(opts, "gemvt");
      if (v.image) c.arg(k, 0, img); else c.arg(k, 0, ctb);
      c.arg(k, 1, stb); c.arg(k, 2, xb); c.arg(k, 3, yb); c.arg(k, 4, K); c.arg(k, 5, N); c.arg(k, 6, global);
      const size_t l0 = std::min<size_t>(64, 256 / v.ks), fibers = size_t(N + v.rows - 1) / v.rows;
      std::vector<double> t;
      for (int i = 0; i < 9; i++) t.push_back(c.run2(k, (fibers + l0 - 1) / l0 * l0, size_t(v.ks), l0, size_t(v.ks)));
      return median(t);
    };
    const size_t const_max = c.num<cl_ulong>(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE);
    double best = 1e9;
    V bv{};
    for (bool image : {false, true})
      for (bool xconst : {false, true})
        for (int rows : {1, 2, 4})
          for (int ks : {1, 2, 4, 8}) {
            if ((image && !img) || (xconst && size_t(K) * 4 > const_max)) continue;
            V v{image, xconst, rows, ks};
            double t = time(v, 1);
            std::printf("gemvt %s %s %.3f ms, %.1f GB/s (%.2fx gemv4m), err/rms %.1e\n", sh.name, label(v).c_str(), t * 1e3,
                        bytes / t / 1e9, tb / t, error(1));
            if (t < best) best = t, bv = v;
          }
    if (!img) std::printf("gemvt %s image1d_buffer unavailable (%zu texels, max %zu)\n", sh.name, size_t(N) * nb, img_max);
    std::printf("gemvt %s best %s %.3f ms, %.1f GB/s\n", sh.name, label(bv).c_str(), best * 1e3, bytes / best / 1e9);
    for (int M : {2, 4, 8, 16}) {
      V v = bv;
      if (v.xconst && size_t(M) * K * 4 > const_max) v.xconst = false;
      double t = time(v, M);
      std::printf("gemvt %s %s M=%d %.3f ms (%.2fx the best M=1 time), err/rms %.1e\n", sh.name, label(v).c_str(), M,
                  t * 1e3, t / best, error(M));
    }
    if (img) p_clReleaseMemObject(img);
    for (cl_mem b : {cb, sb, ctb, stb, xb, yb}) p_clReleaseMemObject(b);
  }
}

uint16_t to_half(float f) {   // round to nearest even, no NaN handling (the data is finite and small)
  uint32_t x;
  std::memcpy(&x, &f, 4);
  uint32_t sign = (x >> 16) & 0x8000;
  int e = int((x >> 23) & 255) - 127 + 15;
  uint32_t m = x & 0x7fffff;
  if (e >= 31) return uint16_t(sign | 0x7c00);
  if (e <= 0) {   // subnormal
    if (e < -10) return uint16_t(sign);
    m |= 0x800000;
    uint32_t shift = uint32_t(14 - e), h = m >> shift, rest = m & ((1u << shift) - 1), halfway = 1u << (shift - 1);
    if (rest > halfway || (rest == halfway && (h & 1))) h++;
    return uint16_t(sign | h);
  }
  uint32_t h = (uint32_t(e) << 10) | (m >> 13), rest = m & 0x1fff;
  if (rest > 0x1000 || (rest == 0x1000 && (h & 1))) h++;
  return uint16_t(sign | h);
}
float from_half(uint16_t h) {
  uint32_t sign = uint32_t(h & 0x8000) << 16, e = (h >> 10) & 31, m = h & 1023, x;
  if (e == 0) {
    float v = std::ldexp(float(m), -24);
    return sign ? -v : v;
  }
  x = sign | ((e - 15 + 127) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}

// the prefill GEMM prototype: NVFP4 7168 x 2560 against M activation rows, FP16 where the device has it
void gemm(Cl& c) {
  const bool f16 = c.has("cl_khr_fp16");
  const int N = 7168, K = 2560;
  std::mt19937 rng(11);
  std::vector<uint8_t> codes(size_t(N) * K / 2), scales(size_t(N) * K / 16);
  for (auto& v : codes) v = uint8_t(rng());
  for (auto& v : scales) v = uint8_t(0x30 + rng() % 16);
  const float global = 0.01f;
  cl_mem cb = c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, codes.size(), codes.data());
  cl_mem sb = c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, scales.size(), scales.data());
  cl_kernel k = c.kernel(f16 ? "" : "", "gemm_t");
  for (int M : {512, 1024}) {
    std::normal_distribution<float> nd(0.f, 0.02f);
    std::vector<float> a(size_t(M) * K);
    std::vector<uint16_t> ah(f16 ? a.size() : 0);
    for (size_t i = 0; i < a.size(); i++) {
      a[i] = nd(rng);
      if (f16) {
        ah[i] = to_half(a[i]);
        a[i] = from_half(ah[i]);   // the reference uses what the GPU sees
      }
    }
    cl_mem ab = f16 ? c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ah.size() * 2, ah.data())
                    : c.buffer(CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, a.size() * 4, a.data());
    std::vector<float> out(size_t(M) * N);
    cl_mem ob = c.buffer(CL_MEM_WRITE_ONLY, out.size() * 4);
    c.arg(k, 0, ab); c.arg(k, 1, cb); c.arg(k, 2, sb); c.arg(k, 3, ob);
    c.arg(k, 4, M); c.arg(k, 5, N); c.arg(k, 6, K); c.arg(k, 7, global);
    size_t gsz[2] = {size_t(N / 128) * 256, size_t((M + 63) / 64)}, lsz[2] = {256, 1};
    std::vector<double> ts;
    for (int i = 0; i < 6; i++) {
      cl_event ev;
      CK(p_clEnqueueNDRangeKernel(c.q, k, 2, nullptr, gsz, lsz, 0, nullptr, &ev));
      CK(p_clWaitForEvents(1, &ev));
      cl_ulong t0 = 0, t1 = 0;
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr);
      p_clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr);
      p_clReleaseEvent(ev);
      if (i) ts.push_back(double(t1 - t0) * 1e-9);
    }
    double t = median(ts);
    CK(p_clEnqueueReadBuffer(c.q, ob, CL_TRUE, 0, out.size() * 4, out.data(), 0, nullptr, nullptr));
    double err = 0, ref2 = 0;
    int checked = 0;
    for (int m : {0, M / 2 + 3, M - 1})
      for (int n = 0; n < N; n += 523) {
        double r = 0;
        for (int kk = 0; kk < K; kk++) {
          uint8_t b = codes[size_t(n) * K / 2 + kk / 2];
          int code = (kk & 1) ? b >> 4 : b & 15;
          r += double(xtg::kE2m1[code]) * xtg::e4m3_to_f32(scales[size_t(n) * K / 16 + kk / 16]) * a[size_t(m) * K + kk];
        }
        r *= global;
        err = std::max(err, std::fabs(r - out[size_t(m) * N + n]));
        ref2 += r * r;
        checked++;
      }
    double rms = std::sqrt(ref2 / checked), tol = f16 ? 5e-3 : 1e-4;
    std::printf("gemm_%s M=%d N=%d K=%d %.2f ms, %.0f GFLOP/s, err/rms %.1e%s\n", f16 ? "fp16" : "fp32", M, N, K, t * 1e3,
                2.0 * M * N * K / t / 1e9, err / rms, err / rms < tol ? "" : "  MISMATCH");
    p_clReleaseMemObject(ab);
    p_clReleaseMemObject(ob);
  }
  p_clReleaseMemObject(cb);
  p_clReleaseMemObject(sb);
}

void peaks(Cl& c) {
  const size_t global = size_t(c.num<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS)) * 2048 * 4, local = 128;
  cl_mem out = c.buffer(CL_MEM_READ_WRITE, global * 4);
  struct P { const char* name; const char* need; double ops; };   // ops per work-item
  for (P p : {P{"peak_fp32", "", 1024 * 4 * 4 * 2.0}, P{"peak_fp16", "cl_khr_fp16", 1024 * 4 * 8 * 2.0},
              P{"peak_int8dot", "cl_khr_integer_dot_product", 1024 * 8 * 4 * 2.0}}) {
    if (*p.need && !c.has(p.need)) {
      std::printf("%s skipped (no %s)\n", p.name, p.need);
      continue;
    }
    cl_kernel k = c.kernel("-cl-std=CL3.0", p.name);
    c.arg(k, 0, out);
    if (std::string(p.name) == "peak_int8dot") {
      cl_uint a = 0x01020304u, b = 0x05060708u;
      c.arg(k, 1, a);
      c.arg(k, 2, b);
    } else {
      float a = 0.999f, b = 0.001f;
      c.arg(k, 1, a);
      c.arg(k, 2, b);
    }
    c.run(k, global, local);
    std::vector<double> ts;
    for (int i = 0; i < 5; i++) ts.push_back(c.run(k, global, local));
    double t = median(ts);
    std::printf("%s %.2f T%s/s\n", p.name, global * p.ops / t / 1e12, std::string(p.name) == "peak_int8dot" ? "OP" : "FLOP");
  }
  p_clReleaseMemObject(out);
}
}  // namespace

int main(int argc, char** argv) {
  Cl c;
  c.init();
  std::printf("device %s | %s | driver %s | %u CUs | local %llu KiB | global %.1f GiB | max alloc %.2f GiB\n",
              c.str(CL_DEVICE_NAME).c_str(), c.str(CL_DEVICE_VERSION).c_str(), c.str(CL_DRIVER_VERSION).c_str(),
              c.num<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS), (unsigned long long)(c.num<cl_ulong>(CL_DEVICE_LOCAL_MEM_SIZE) >> 10),
              c.num<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE) / 1073741824.0, c.num<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE) / 1073741824.0);
  for (const char* e : {"cl_qcom_recordable_queues", "cl_khr_integer_dot_product", "cl_qcom_dot_product8", "cl_khr_fp16",
                        "cl_khr_subgroups", "cl_qcom_reqd_sub_group_size", "cl_qcom_perf_hint", "cl_qcom_ml_ops",
                        "cl_qcom_ext_host_ptr", "cl_qcom_ion_host_ptr", "cl_qcom_dmabuf_host_ptr",
                        "cl_qcom_android_ahardwarebuffer_host_ptr", "cl_khr_command_buffer", "cl_qcom_onchip_global_memory"})
    std::printf("extension %s %s\n", e, c.has(e) ? "yes" : "no");
  std::printf("extensions %s\n", c.ext.c_str());
  std::string only = argc > 1 ? argv[1] : "";
  if (only.empty() || only == "launch") launch(c);
  if (only.empty() || only == "read") bandwidth(c);
  if (only.empty() || only == "gemv") gemv(c);
  if (only.empty() || only == "gemvt") gemvt(c);
  if (only.empty() || only == "gemm") gemm(c);
  if (only.empty() || only == "peak") peaks(c);
  std::printf("done\n");
  return 0;
}
