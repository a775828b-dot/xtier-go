// gpu_probe: the building blocks of a GPU-resident Dory step, measured on the device (adb shell; also runs on any
// OpenCL GPU for a correctness check). Prints one line per measurement:
//   device / extensions          what the driver offers (recordable queues, integer dot product, subgroups, fp16)
//   launch async|sync|roundtrip  microseconds per kernel: queued back to back, clFinish after each, and the
//                                current backend's pattern (upload 10 KiB, kernel, read 28 KiB, clFinish)
//   read device|hostptr          streaming read bandwidth of a 512 MiB buffer, driver-owned and ALLOC_HOST_PTR
//   gemv4 M=1..16                NVFP4 7168 x 2560 against M activation rows at once (what speculative decoding
//                                needs): time per call and the weight bytes per second
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
  X(clWaitForEvents) X(clReleaseEvent)
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
  if (only.empty() || only == "peak") peaks(c);
  std::printf("done\n");
  return 0;
}
