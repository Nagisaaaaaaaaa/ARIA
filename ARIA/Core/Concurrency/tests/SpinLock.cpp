#include "ARIA/Concurrency/SpinLock.h"

#include <gtest/gtest.h>

#include <mutex>
#include <thread>

namespace ARIA {

namespace {

static constinit SpinLock lock{};
static constinit size_t count = 0;

void Increase0(size_t n) {
  for (size_t i = 0; i < n; ++i) {
    std::lock_guard guard(lock);
    ++count;
  }
}

void Increase1(size_t n) {
  std::lock_guard guard(lock);
  for (size_t i = 0; i < n; ++i) {
    ++count;
  }
}

} // namespace

TEST(SpinLock, Base) {
  {
    std::vector<std::jthread> threads(10);

    for (size_t i = 0; i < threads.size(); ++i) {
      auto &t = threads[i];
      t = std::jthread{Increase0, i * 1000};
    }

    for (auto &t : threads) {
      t.join();
    }

    EXPECT_EQ(count, 45000);
  }

  {
    std::vector<std::jthread> threads(10);

    for (size_t i = 0; i < threads.size(); ++i) {
      auto &t = threads[i];
      t = std::jthread{Increase1, i * 1000};
    }

    for (auto &t : threads) {
      t.join();
    }

    EXPECT_EQ(count, 90000);
  }
}

} // namespace ARIA
