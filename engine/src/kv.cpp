#include "xtg/kv.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <limits>

#include "xtg/common.h"
#include "xtg/fp.h"

namespace xtg {

const char* kv_dtype_name(KvDtype d) { return d == KvDtype::FP8 ? "fp8" : "bf16"; }

// FP32 -> E4M3 with the bit operations of the CUDA conversion (round to nearest even, saturating to 448).
static inline uint8_t e4m3_bits(float x) {
  uint32_t bits;
  std::memcpy(&bits, &x, 4);
  uint32_t sign = (bits >> 24) & 0x80, a = bits & 0x7fffffff;
  uint32_t code;
  if (a > 0x7f800000) return uint8_t(sign | 0x7f);
  if (a < 0x3c800000) {   // below 2^-6: subnormal steps of 2^-9, rounded to the nearest even
    float t;
    std::memcpy(&t, &a, 4);
    t *= 512.f;
    float lo = std::floor(t);
    uint32_t steps = uint32_t(lo);
    float half = t - lo;
    steps += (half > 0.5f) || (half == 0.5f && (steps & 1));
    code = steps;
  } else {
    uint32_t r = a + 0x7ffff + ((a >> 20) & 1);
    code = std::min<uint32_t>((((r >> 23) - 120) << 3) | ((r >> 20) & 7), 0x7e);
  }
  return uint8_t(sign | code);
}

KvCache::KvCache(int kv_heads, int head_dim, int64_t slots, int ring, KvDtype dtype)
    : kv_heads_(kv_heads), head_dim_(head_dim), slots_(ring > 0 ? ring : slots), ring_(ring > 0), dtype_(dtype) {
  XTG_CHECK(slots_ > 0, "KV cache with no slots");
  elem_ = dtype == KvDtype::FP8 ? 1 : 2;
  size_t row = size_t(kv_heads) * size_t(head_dim) * elem_;
  size_t kbytes = size_t(slots_) * row;
  size_t sbytes = dtype == KvDtype::FP8 ? size_t(slots_) * size_t(kv_heads) * sizeof(float) : 0;
  size_t page = size_t(::sysconf(_SC_PAGESIZE));
  auto up = [&](size_t n) { return (n + page - 1) / page * page; };
  bytes_ = 2 * up(kbytes) + 2 * up(sbytes);
  void* p = ::mmap(nullptr, bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
  XTG_CHECK(p != MAP_FAILED, "cannot reserve %zu bytes of KV cache: %s", bytes_, std::strerror(errno));
  base_ = static_cast<uint8_t*>(p);
  k_ = base_;
  v_ = k_ + up(kbytes);
  if (sbytes) {
    ks_ = reinterpret_cast<float*>(v_ + up(kbytes));
    vs_ = reinterpret_cast<float*>(reinterpret_cast<uint8_t*>(ks_) + up(sbytes));
  }
}

KvCache::~KvCache() {
  if (base_) ::munmap(base_, bytes_);
}

size_t KvCache::bytes_per_position() const {
  return 2 * size_t(kv_heads_) * (size_t(head_dim_) * elem_ + (dtype_ == KvDtype::FP8 ? sizeof(float) : 0));
}

size_t KvCache::used_bytes(int64_t n_past) const {
  return size_t(ring_ ? slots_ : std::min(n_past, slots_)) * bytes_per_position();
}

void KvCache::store(int64_t pos, int n, const float* k, const float* v) {
  int64_t first = pos;
  if (ring_ && n > slots_) first = pos + n - slots_;
  XTG_CHECK(ring_ || pos + n <= slots_, "KV cache full: position %lld of %lld", (long long)(pos + n), (long long)slots_);
  const size_t hd = size_t(head_dim_);
  for (int64_t p = first; p < pos + n; ++p) {
    int64_t slot = ring_ ? p % slots_ : p;
    for (int h = 0; h < kv_heads_; ++h) {
      const float* src_k = k + (size_t(p - pos) * size_t(kv_heads_) + size_t(h)) * hd;
      const float* src_v = v + (size_t(p - pos) * size_t(kv_heads_) + size_t(h)) * hd;
      size_t idx = size_t(slot) * size_t(kv_heads_) + size_t(h);
      if (dtype_ == KvDtype::BF16) {
        uint16_t* dk = reinterpret_cast<uint16_t*>(k_) + idx * hd;
        uint16_t* dv = reinterpret_cast<uint16_t*>(v_) + idx * hd;
        for (size_t j = 0; j < hd; ++j) {
          dk[j] = f32_to_bf16(src_k[j]);
          dv[j] = f32_to_bf16(src_v[j]);
        }
      } else {
        float ak = 0.f, av = 0.f;
        for (size_t j = 0; j < hd; ++j) {
          ak = std::max(ak, std::fabs(src_k[j]));
          av = std::max(av, std::fabs(src_v[j]));
        }
        float sk = ak > 0.f ? ak / 448.f : 1.f, sv = av > 0.f ? av / 448.f : 1.f;
        float ik = 1.f / sk, iv = 1.f / sv;
        uint8_t* dk = k_ + idx * hd;
        uint8_t* dv = v_ + idx * hd;
        for (size_t j = 0; j < hd; ++j) {
          dk[j] = e4m3_bits(src_k[j] * ik);
          dv[j] = e4m3_bits(src_v[j] * iv);
        }
        ks_[idx] = sk;
        vs_[idx] = sv;
      }
    }
  }
}

void KvCache::load(int h, int64_t p0, int64_t p1, float* k, float* v) const {
  const size_t hd = size_t(head_dim_);
  for (int64_t p = p0; p < p1; ++p) {
    int64_t slot = ring_ ? p % slots_ : p;
    size_t idx = size_t(slot) * size_t(kv_heads_) + size_t(h);
    float* ok = k + size_t(p - p0) * hd;
    float* ov = v + size_t(p - p0) * hd;
    if (dtype_ == KvDtype::BF16) {
      const uint16_t* sk = reinterpret_cast<const uint16_t*>(k_) + idx * hd;
      const uint16_t* sv = reinterpret_cast<const uint16_t*>(v_) + idx * hd;
      for (size_t j = 0; j < hd; ++j) {
        ok[j] = bf16_to_f32(sk[j]);
        ov[j] = bf16_to_f32(sv[j]);
      }
    } else {
      const uint8_t* sk = k_ + idx * hd;
      const uint8_t* sv = v_ + idx * hd;
      float fk = ks_[idx], fv = vs_[idx];
      for (size_t j = 0; j < hd; ++j) {
        ok[j] = e4m3_to_f32(sk[j]) * fk;
        ov[j] = e4m3_to_f32(sv[j]) * fv;
      }
    }
  }
}

void KvCache::save(std::vector<uint8_t>& out) const {
  out.assign(base_, base_ + bytes_);
}

void KvCache::restore(const std::vector<uint8_t>& in) {
  XTG_CHECK(in.size() == bytes_, "KV snapshot of %zu bytes for a cache of %zu", in.size(), bytes_);
  std::memcpy(base_, in.data(), bytes_);
}

void KvCache::release() { ::madvise(base_, bytes_, MADV_DONTNEED); }

// --- attention ---

namespace {
constexpr int kRows = 8;     // query rows per task
constexpr int kTile = 64;    // keys decoded per step
constexpr int kMaxGroup = 8; // query heads per KV head

struct Partial {
  float m[kRows][kMaxGroup];
  float l[kRows][kMaxGroup];
};
}  // namespace

float dot_f32(const float* a, const float* b, int k);   // kernels.cpp

void attention(ThreadPool& pool, const AttnArgs& a) {
  const int group = a.heads / a.kv_heads, d = a.head_dim;
  XTG_CHECK(group * a.kv_heads == a.heads && group <= kMaxGroup, "attention: %d heads over %d KV heads", a.heads,
            a.kv_heads);
  auto lo_of = [&](int64_t t) { return a.window > 0 ? std::max<int64_t>(0, t - a.window + 1) : int64_t(0); };
  const int64_t k_lo = lo_of(a.pos0), k_hi = a.pos0 + a.n_q;
  const int64_t span = k_hi - k_lo;
  const int qblocks = (a.n_q + kRows - 1) / kRows;
  int splits = 1;
  if (a.n_q <= 4 && span > 2 * kTile)
    splits = int(std::min<int64_t>(std::max(1, 2 * pool.size() / a.kv_heads), (span + 1023) / 1024));
  splits = std::max(1, splits);
  const int64_t tasks = int64_t(a.kv_heads) * qblocks * splits;
  const size_t acc_per_task = size_t(kRows) * size_t(group) * size_t(d);
  std::vector<Partial> parts(splits > 1 ? size_t(tasks) : 0);
  std::vector<float> accs(splits > 1 ? size_t(tasks) * acc_per_task : 0);
  const float neg_inf = -std::numeric_limits<float>::infinity();

  pool.parallel_for(tasks, 1, [&](int64_t t0, int64_t t1, int) {
    thread_local std::vector<float> kt, vt, acc, sc;
    kt.resize(size_t(kTile) * size_t(d));
    vt.resize(size_t(kTile) * size_t(d));
    acc.resize(acc_per_task);
    sc.resize(kTile);
    for (int64_t task = t0; task < t1; ++task) {
      int s = int(task % splits);
      int qb = int((task / splits) % qblocks);
      int h = int(task / (int64_t(splits) * qblocks));
      int i0 = qb * kRows, i1 = std::min(a.n_q, i0 + kRows);
      // keys this task covers: its share of the span, cut to what its rows see
      int64_t s0 = k_lo + span * s / splits, s1 = k_lo + span * (s + 1) / splits;
      int64_t r0 = std::max(s0, lo_of(a.pos0 + i0)), r1 = std::min(s1, a.pos0 + i1);
      Partial st;
      for (int i = 0; i < kRows; ++i)
        for (int g = 0; g < kMaxGroup; ++g) {
          st.m[i][g] = neg_inf;
          st.l[i][g] = 0.f;
        }
      std::fill(acc.begin(), acc.end(), 0.f);
      for (int64_t kp = r0; kp < r1; kp += kTile) {
        int64_t kq = std::min(r1, kp + kTile);
        // the tile's rows: cached positions, then this call's own
        int64_t cut = std::min(kq, std::max(kp, a.pos0));
        if (cut > kp) a.cache->load(h, kp, cut, kt.data(), vt.data());
        for (int64_t p = cut; p < kq; ++p) {
          const size_t src = (size_t(p - a.pos0) * size_t(a.kv_heads) + size_t(h)) * size_t(d);
          std::memcpy(kt.data() + size_t(p - kp) * size_t(d), a.k + src, size_t(d) * sizeof(float));
          std::memcpy(vt.data() + size_t(p - kp) * size_t(d), a.v + src, size_t(d) * sizeof(float));
        }
        for (int i = i0; i < i1; ++i) {
          int64_t t = a.pos0 + i;
          int64_t j0 = std::max(kp, lo_of(t)), j1 = std::min(kq, t + 1);
          if (j0 >= j1) continue;
          for (int g = 0; g < group; ++g) {
            const float* qv = a.q + (size_t(i) * size_t(a.heads) + size_t(h * group + g)) * size_t(d);
            float mx = neg_inf;
            for (int64_t j = j0; j < j1; ++j) {
              const float* kv = kt.data() + size_t(j - kp) * size_t(d);
              float dot = dot_f32(qv, kv, d) * a.scale;
              sc[size_t(j - j0)] = dot;
              mx = std::max(mx, dot);
            }
            float& m = st.m[i - i0][g];
            float& l = st.l[i - i0][g];
            float m_new = std::max(m, mx);
            float corr = m == neg_inf ? 0.f : std::exp(m - m_new);
            float* ac = acc.data() + (size_t(i - i0) * size_t(group) + size_t(g)) * size_t(d);
            if (corr != 1.f)
              for (int e = 0; e < d; ++e) ac[e] *= corr;
            l *= corr;
            for (int64_t j = j0; j < j1; ++j) {
              float w = std::exp(sc[size_t(j - j0)] - m_new);
              l += w;
              const float* vv = vt.data() + size_t(j - kp) * size_t(d);
              for (int e = 0; e < d; ++e) ac[e] += w * vv[e];
            }
            m = m_new;
          }
        }
      }
      if (splits == 1) {
        for (int i = i0; i < i1; ++i)
          for (int g = 0; g < group; ++g) {
            float* o = a.out + (size_t(i) * size_t(a.heads) + size_t(h * group + g)) * size_t(d);
            const float* ac = acc.data() + (size_t(i - i0) * size_t(group) + size_t(g)) * size_t(d);
            float l = st.l[i - i0][g];
            float inv = l > 0.f ? 1.f / l : 0.f;
            for (int e = 0; e < d; ++e) o[e] = ac[e] * inv;
          }
      } else {
        parts[size_t(task)] = st;
        std::memcpy(accs.data() + size_t(task) * acc_per_task, acc.data(), acc_per_task * sizeof(float));
      }
    }
  });
  if (splits == 1) return;
  // combine the splits of each (KV head, query block)
  for (int h = 0; h < a.kv_heads; ++h)
    for (int qb = 0; qb < qblocks; ++qb) {
      int i0 = qb * kRows, i1 = std::min(a.n_q, i0 + kRows);
      int64_t base = (int64_t(h) * qblocks + qb) * splits;
      for (int i = i0; i < i1; ++i)
        for (int g = 0; g < group; ++g) {
          float mx = neg_inf;
          for (int s = 0; s < splits; ++s) mx = std::max(mx, parts[size_t(base + s)].m[i - i0][g]);
          float* o = a.out + (size_t(i) * size_t(a.heads) + size_t(h * group + g)) * size_t(d);
          std::fill(o, o + d, 0.f);
          float l = 0.f;
          for (int s = 0; s < splits; ++s) {
            const Partial& p = parts[size_t(base + s)];
            if (p.l[i - i0][g] == 0.f) continue;
            float w = std::exp(p.m[i - i0][g] - mx);
            l += w * p.l[i - i0][g];
            const float* ac = accs.data() + size_t(base + s) * acc_per_task + (size_t(i - i0) * size_t(group) + size_t(g)) * size_t(d);
            for (int e = 0; e < d; ++e) o[e] += w * ac[e];
          }
          float inv = l > 0.f ? 1.f / l : 0.f;
          for (int e = 0; e < d; ++e) o[e] *= inv;
        }
    }
}

}  // namespace xtg
