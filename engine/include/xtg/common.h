// Errors, logging and timing shared by the engine, the model packs and the server.
#pragma once

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <string>

namespace xtg {

// Inside the engine an error is an exception; the C API and the HTTP server turn it into a message.
struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

[[noreturn]] void fail(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

// One line to stderr (and to logcat on Android), prefixed with the time.
void log(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

inline double now_s() {
  return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string format(const char* fmt, ...) __attribute__((format(printf, 1, 2)));

constexpr const char* kVersion = "0.0.1";

}  // namespace xtg

#define XTG_CHECK(cond, ...)       \
  do {                             \
    if (!(cond)) ::xtg::fail(__VA_ARGS__); \
  } while (0)
