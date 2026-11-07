// A safetensors file mapped read-only: the engine computes on the tensors where they lie (no copy, no conversion),
// so loading is a map plus a read-ahead and the page cache can drop them under memory pressure.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace xtg {

struct TensorView {
  std::string dtype;            // "BF16", "F32", "U8", "F8_E4M3", ...
  std::vector<int64_t> shape;
  const uint8_t* data = nullptr;
  size_t nbytes = 0;

  int64_t numel() const;
  std::string shape_str() const;
};

class SafeTensors {
 public:
  explicit SafeTensors(const std::string& path);
  ~SafeTensors();
  SafeTensors(const SafeTensors&) = delete;
  SafeTensors& operator=(const SafeTensors&) = delete;

  bool has(const std::string& name) const { return tensors_.count(name) != 0; }
  // Throws naming the tensor when it is missing or not of this dtype / shape.
  const TensorView& get(const std::string& name) const;
  const TensorView& get(const std::string& name, const char* dtype, const std::vector<int64_t>& shape) const;
  const std::map<std::string, TensorView>& tensors() const { return tensors_; }
  const std::string& path() const { return path_; }
  size_t file_size() const { return size_; }

  // Ask the kernel to read the file ahead (madvise WILLNEED); touch() reads every page now.
  void advise_willneed() const;
  void touch() const;

 private:
  std::string path_;
  int fd_ = -1;
  size_t size_ = 0;
  uint8_t* base_ = nullptr;
  std::map<std::string, TensorView> tensors_;
};

}  // namespace xtg
