// A fixed pool of worker threads for the kernels. One parallel region at a time (the engine runs one forward at a
// time); idle workers spin for a while (a decode step is a few hundred short regions) and then sleep.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace xtg {

class ThreadPool {
 public:
  // n_threads counts the calling thread. cpus: optional affinity, cpus[i] for thread i (the caller is thread 0).
  explicit ThreadPool(int n_threads, std::vector<int> cpus = {});
  ~ThreadPool();
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  int size() const { return n_; }

  // fn(begin, end, thread) over [0, n) in pieces of `grain`, handed out dynamically (the cores of a phone run at
  // different clocks). Runs inline when there is one piece.
  void parallel_for(int64_t n, int64_t grain, const std::function<void(int64_t, int64_t, int)>& fn);

  // Workers idle longer than this go to sleep (0: sleep at once).
  void set_spin_us(int us) { spin_us_ = us; }

 private:
  void worker(int index);
  void work(int thread);

  int n_;
  std::vector<int> cpus_;
  std::vector<std::thread> threads_;
  std::atomic<uint64_t> generation_{0};
  std::atomic<int64_t> next_{0};
  std::atomic<int> pending_{0};
  std::atomic<bool> stop_{false};
  std::atomic<int> sleepers_{0};
  int spin_us_ = 2000;
  const std::function<void(int64_t, int64_t, int)>* fn_ = nullptr;
  int64_t n_items_ = 0, grain_ = 1;
  std::mutex mutex_;
  std::condition_variable wake_;
};

// Pin the calling thread to one CPU (no-op when cpu < 0). False when the system refused.
bool pin_thread(int cpu);

}  // namespace xtg
