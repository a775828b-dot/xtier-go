#include "xtg/common.h"

#include <cstdarg>
#include <cstdio>
#include <ctime>
#include <mutex>
#include <vector>

#ifdef __ANDROID__
#include <android/log.h>
#endif

namespace xtg {

static std::string vformat(const char* fmt, va_list ap) {
  va_list copy;
  va_copy(copy, ap);
  int n = std::vsnprintf(nullptr, 0, fmt, copy);
  va_end(copy);
  std::vector<char> buf(size_t(n > 0 ? n : 0) + 1);
  std::vsnprintf(buf.data(), buf.size(), fmt, ap);
  return std::string(buf.data(), size_t(n > 0 ? n : 0));
}

std::string format(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::string s = vformat(fmt, ap);
  va_end(ap);
  return s;
}

void fail(const char* fmt, ...) {
  va_list ap;
  va_start(ap, fmt);
  std::string s = vformat(fmt, ap);
  va_end(ap);
  throw Error(s);
}

void log(const char* fmt, ...) {
  static std::mutex mutex;
  va_list ap;
  va_start(ap, fmt);
  std::string s = vformat(fmt, ap);
  va_end(ap);
#ifdef __ANDROID__
  __android_log_write(ANDROID_LOG_INFO, "xtier-go", s.c_str());
#endif
  std::time_t t = std::time(nullptr);
  char stamp[32];
  std::strftime(stamp, sizeof(stamp), "%m-%d %H:%M:%S", std::localtime(&t));
  std::lock_guard<std::mutex> lock(mutex);
  std::fprintf(stderr, "%s [xtg] %s\n", stamp, s.c_str());
  std::fflush(stderr);
}

}  // namespace xtg
