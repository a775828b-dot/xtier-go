// JNI bridge: io.github.a775828b.xtiergo.Native -> the C API (server/xtg_capi.h).
#include <jni.h>

#include <string>
#include <vector>

#include "xtg_capi.h"

static std::string str(JNIEnv* env, jstring s) {
  if (!s) return {};
  const char* c = env->GetStringUTFChars(s, nullptr);
  std::string out(c);
  env->ReleaseStringUTFChars(s, c);
  return out;
}

static jstring jstr(JNIEnv* env, const std::string& s) { return env->NewStringUTF(s.c_str()); }

static std::string grow(int (*fn)(char*, int)) {
  std::vector<char> buf(16384);
  int n = fn(buf.data(), int(buf.size()));
  if (n >= int(buf.size())) {
    buf.resize(size_t(n) + 1);
    fn(buf.data(), int(buf.size()));
  }
  return std::string(buf.data());
}

extern "C" JNIEXPORT jstring JNICALL Java_io_github_a775828b_xtiergo_Native_start(JNIEnv* env, jobject, jstring config) {
  char err[2048] = {0};
  if (xtg_start(str(env, config).c_str(), err, sizeof(err)) == 0) return nullptr;
  return jstr(env, err);
}

extern "C" JNIEXPORT void JNICALL Java_io_github_a775828b_xtiergo_Native_stop(JNIEnv*, jobject) { xtg_stop(); }

extern "C" JNIEXPORT jstring JNICALL Java_io_github_a775828b_xtiergo_Native_status(JNIEnv* env, jobject) {
  return jstr(env, grow(xtg_status));
}

extern "C" JNIEXPORT jstring JNICALL Java_io_github_a775828b_xtiergo_Native_estimate(JNIEnv* env, jobject, jstring dir,
                                                                                     jstring options) {
  std::vector<char> buf(8192);
  std::string d = str(env, dir), o = str(env, options);
  int n = xtg_estimate(d.c_str(), o.c_str(), buf.data(), int(buf.size()));
  if (n >= int(buf.size())) {
    buf.resize(size_t(n) + 1);
    xtg_estimate(d.c_str(), o.c_str(), buf.data(), int(buf.size()));
  }
  return jstr(env, buf.data());
}

extern "C" JNIEXPORT jstring JNICALL Java_io_github_a775828b_xtiergo_Native_version(JNIEnv* env, jobject) {
  return jstr(env, xtg_version());
}
