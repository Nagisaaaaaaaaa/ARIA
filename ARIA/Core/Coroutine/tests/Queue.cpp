#include "ARIA/Coroutine/Queue.h"

#include <cppcoro/static_thread_pool.hpp>
#include <cppcoro/sync_wait.hpp>
#include <cppcoro/task.hpp>
#include <cppcoro/when_all.hpp>
#include <gtest/gtest.h>

#include <queue>

namespace ARIA {

namespace {

class MySpinLock {
public:
  void lock() noexcept {
    for (;;) {
      if (!lock_.exchange(true, std::memory_order_acquire)) {
        return;
      }
      while (lock_.load(std::memory_order_relaxed)) {
#if ARIA_ICC || ARIA_MSVC
        _mm_pause();
#else
        __builtin_ia32_pause();
#endif
      }
    }
  }

  void unlock() noexcept { lock_.store(false, std::memory_order_release); }

  bool try_lock() noexcept {
    return !lock_.load(std::memory_order_relaxed) && !lock_.exchange(true, std::memory_order_acquire);
  }

private:
  std::atomic<bool> lock_{false};

public:
  MySpinLock() = default;

  ARIA_COPY_MOVE_ABILITY(MySpinLock, delete, delete);
};

//
//
//
static constinit std::atomic<int> v = 0;

static cppcoro::static_thread_pool threadPool;
static Coroutine::queue<MySpinLock> queue;
static Coroutine::queue<> queueNonThreadSafe;

//
//
//
cppcoro::task<> Func0() {
  co_await threadPool.schedule();
  ++v;
  co_await queue.schedule();
  --v;
}

cppcoro::task<> Func1() {
  int success = 0;
  for (int i = 0; i < 500; ++i)
    success += static_cast<int>(queue.try_pop());

  while (v != 1000 - success) {}

  // Wait until all the coroutines having arrived at the queue.
  // 0.5s is enough.
  using namespace std::chrono_literals;
  std::this_thread::sleep_for(0.5s);

  while (queue.try_pop()) {}
  EXPECT_EQ(v, 0);
  co_return;
}

} // namespace

//
//
//
TEST(Queue, Base) {
  std::vector<cppcoro::task<>> tasks;
  for (int i = 0; i < 1000; ++i)
    tasks.emplace_back(Func0());
  tasks.emplace_back(Func1());
  cppcoro::sync_wait(cppcoro::when_all(std::move(tasks)));
}

} // namespace ARIA
