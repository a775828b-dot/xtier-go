#pragma once

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xtg/kernels.h"

struct _cl_mem;
struct _cl_kernel;

namespace xtg {

// One model owns its immutable device weights and queue.
//
// Two ways to use it. matmul(..., const float* x, ...) is synchronous: upload, compute, read back (the hybrid path
// where the session keeps activations, attention and the rest on the CPU). The device-resident calls (alloc, write,
// launch, matmul on Mem buffers, ...) only queue work on the one in-order queue; nothing waits until read() or
// finish(), so a whole forward step runs without a host round trip per layer.
class OpenClBackend {
 public:
  using Mem = _cl_mem*;
  using Kernel = _cl_kernel*;
  using Range = std::array<size_t, 2>;

  OpenClBackend();
  ~OpenClBackend();
  OpenClBackend(const OpenClBackend&) = delete;
  OpenClBackend& operator=(const OpenClBackend&) = delete;
  void prepare(const Nvfp4Weight& w);
  void prepare(const Bf16Weight& w);
  void matmul(const std::vector<const Nvfp4Weight*>& ws, const float* x, int m, const std::vector<float*>& ys);
  void matmul(const Bf16Weight& w, const float* x, int m, float* y);
  std::string device_name() const;
  size_t weight_bytes() const;

  // device-resident path
  Mem alloc(size_t bytes);
  void release(Mem m);
  void write(Mem m, size_t offset, const void* src, size_t bytes);   // blocking: src may be reused on return
  void read(Mem m, size_t offset, void* dst, size_t bytes);           // blocking: waits for all queued work
  void copy(Mem src, size_t src_offset, Mem dst, size_t dst_offset, size_t bytes);
  void finish();
  // kernels from a program source (built once per source and options)
  Kernel kernel(const char* source, const std::string& options, const char* name);
  template <class... A>
  void launch(Kernel k, Range global, Range local, const A&... args) {
    unsigned i = 0;
    (set_arg(k, i++, sizeof(A), &args), ...);
    enqueue(k, global, local);
  }
  // NVFP4 projections of the same input into consecutive [m][n_out] blocks of y, from y_offset (floats)
  void matmul(const std::vector<const Nvfp4Weight*>& ws, Mem x, int m, Mem y, size_t y_offset);
  void matmul(const Bf16Weight& w, Mem x, int m, Mem y);

 private:
  void set_arg(Kernel k, unsigned index, size_t size, const void* value);
  void enqueue(Kernel k, Range global, Range local);
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace xtg
