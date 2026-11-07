#include "xtg/threadpool.h"

#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>

namespace xtg {

bool pin_thread(int cpu) {
  if (cpu < 0) return true;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return sched_setaffinity(pid_t(syscall(SYS_gettid)), sizeof(set), &set) == 0;
}

static inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

ThreadPool::ThreadPool(int n_threads, std::vector<int> cpus) : n_(std::max(1, n_threads)), cpus_(std::move(cpus)) {
  if (!cpus_.empty()) pin_thread(cpus_[0]);
  for (int i = 1; i < n_; ++i) threads_.emplace_back(&ThreadPool::worker, this, i);
}

ThreadPool::~ThreadPool() {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    stop_.store(true);
    generation_.fetch_add(1);
  }
  wake_.notify_all();
  for (auto& t : threads_) t.join();
}

void ThreadPool::work(int thread) {
  for (;;) {
    int64_t begin = next_.fetch_add(grain_, std::memory_order_relaxed);
    if (begin >= n_items_) break;
    (*fn_)(begin, std::min(begin + grain_, n_items_), thread);
  }
}

void ThreadPool::worker(int index) {
  if (index < int(cpus_.size())) pin_thread(cpus_[index]);
  uint64_t seen = 0;
  for (;;) {
    auto start = std::chrono::steady_clock::now();
    uint64_t g;
    int spins = 0;
    while ((g = generation_.load(std::memory_order_acquire)) == seen) {
      cpu_relax();
      if (++spins < 4096) continue;
      spins = 0;
      auto idle = std::chrono::steady_clock::now() - start;
      if (std::chrono::duration_cast<std::chrono::microseconds>(idle).count() < spin_us_) continue;
      std::unique_lock<std::mutex> lock(mutex_);
      sleepers_.fetch_add(1);
      wake_.wait(lock, [&] { return generation_.load(std::memory_order_acquire) != seen; });
      sleepers_.fetch_sub(1);
    }
    seen = g;
    if (stop_.load()) return;
    work(index);
    pending_.fetch_sub(1, std::memory_order_acq_rel);
  }
}

void ThreadPool::parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t, int)>& fn) {
  if (n <= 0) return;
  grain = std::max<int64_t>(1, grain);
  if (n_ == 1 || n <= grain) {
    fn(0, n, 0);
    return;
  }
  fn_ = &fn;
  n_items_ = n;
  grain_ = grain;
  next_.store(0, std::memory_order_relaxed);
  pending_.store(n_ - 1, std::memory_order_relaxed);
  {
    // Under the lock, so a worker going to sleep cannot miss this generation.
    std::lock_guard<std::mutex> lock(mutex_);
    generation_.fetch_add(1, std::memory_order_release);
  }
  if (sleepers_.load() > 0) wake_.notify_all();
  work(0);
  while (pending_.load(std::memory_order_acquire) > 0) cpu_relax();
  fn_ = nullptr;
}

}  // namespace xtg
