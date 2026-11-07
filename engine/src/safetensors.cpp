#include "xtg/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <nlohmann/json.hpp>

#include "xtg/common.h"

namespace xtg {

int64_t TensorView::numel() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

std::string TensorView::shape_str() const {
  std::string s = "(";
  for (size_t i = 0; i < shape.size(); ++i) s += (i ? ", " : "") + std::to_string(shape[i]);
  return s + ")";
}

static size_t dtype_bytes(const std::string& dtype) {
  if (dtype == "F32" || dtype == "I32" || dtype == "U32") return 4;
  if (dtype == "BF16" || dtype == "F16" || dtype == "I16" || dtype == "U16") return 2;
  if (dtype == "U8" || dtype == "I8" || dtype == "F8_E4M3" || dtype == "F8_E5M2" || dtype == "BOOL") return 1;
  if (dtype == "F64" || dtype == "I64" || dtype == "U64") return 8;
  return 0;
}

SafeTensors::SafeTensors(const std::string& path) : path_(path) {
  fd_ = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
  XTG_CHECK(fd_ >= 0, "cannot open %s: %s", path.c_str(), std::strerror(errno));
  struct stat st;
  XTG_CHECK(::fstat(fd_, &st) == 0, "cannot stat %s", path.c_str());
  size_ = size_t(st.st_size);
  XTG_CHECK(size_ >= 8, "%s: too small for a safetensors file", path.c_str());
  void* p = ::mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
  XTG_CHECK(p != MAP_FAILED, "cannot map %s: %s", path.c_str(), std::strerror(errno));
  base_ = static_cast<uint8_t*>(p);

  uint64_t header_len;
  std::memcpy(&header_len, base_, 8);
  XTG_CHECK(header_len > 0 && 8 + header_len <= size_, "%s: bad header length %llu", path.c_str(),
            (unsigned long long)header_len);
  auto header = nlohmann::json::parse(reinterpret_cast<const char*>(base_ + 8),
                                      reinterpret_cast<const char*>(base_ + 8 + header_len));
  const uint8_t* data = base_ + 8 + header_len;
  size_t data_len = size_ - 8 - header_len;
  for (auto it = header.begin(); it != header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const auto& t = it.value();
    TensorView v;
    v.dtype = t.at("dtype").get<std::string>();
    v.shape = t.at("shape").get<std::vector<int64_t>>();
    auto off = t.at("data_offsets").get<std::vector<uint64_t>>();
    XTG_CHECK(off.size() == 2 && off[0] <= off[1] && off[1] <= data_len, "%s: bad offsets for %s", path.c_str(),
              it.key().c_str());
    v.data = data + off[0];
    v.nbytes = size_t(off[1] - off[0]);
    size_t eb = dtype_bytes(v.dtype);
    XTG_CHECK(eb == 0 || size_t(v.numel()) * eb == v.nbytes, "%s: %s has %zu bytes for shape %s", path.c_str(),
              it.key().c_str(), v.nbytes, v.shape_str().c_str());
    tensors_.emplace(it.key(), std::move(v));
  }
}

SafeTensors::~SafeTensors() {
  if (base_) ::munmap(base_, size_);
  if (fd_ >= 0) ::close(fd_);
}

const TensorView& SafeTensors::get(const std::string& name) const {
  auto it = tensors_.find(name);
  XTG_CHECK(it != tensors_.end(), "%s: no tensor %s", path_.c_str(), name.c_str());
  return it->second;
}

const TensorView& SafeTensors::get(const std::string& name, const char* dtype, const std::vector<int64_t>& shape) const {
  const TensorView& v = get(name);
  XTG_CHECK(v.dtype == dtype, "%s: %s is %s, expected %s", path_.c_str(), name.c_str(), v.dtype.c_str(), dtype);
  XTG_CHECK(v.shape == shape, "%s: %s has shape %s", path_.c_str(), name.c_str(), v.shape_str().c_str());
  return v;
}

void SafeTensors::advise_willneed() const {
  if (base_) ::madvise(base_, size_, MADV_WILLNEED);
}

void SafeTensors::touch() const {
  volatile uint8_t sink = 0;
  long page = ::sysconf(_SC_PAGESIZE);
  for (size_t off = 0; off < size_; off += size_t(page)) sink ^= base_[off];
  (void)sink;
}

}  // namespace xtg
