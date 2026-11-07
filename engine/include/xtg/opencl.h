#pragma once

#include <memory>
#include <string>

#include "xtg/kernels.h"

namespace xtg {

// One model owns its immutable device weights and queue. Calls are synchronous; the
// existing session keeps CPU activations, attention and cancellation boundaries.
class OpenClBackend {
 public:
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

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace xtg
