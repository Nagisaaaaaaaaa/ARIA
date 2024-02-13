#include "ARIA/Registry.h"

#include <gtest/gtest.h>

#include <ranges>
#include <thread>

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

struct A : public Registry<A, MySpinLock> {
  using Registry<A, MySpinLock>::size;
  using Registry<A, MySpinLock>::begin;
  using Registry<A, MySpinLock>::end;
  using Registry<A, MySpinLock>::cbegin;
  using Registry<A, MySpinLock>::cend;
  using Registry<A, MySpinLock>::range;
  using Registry<A, MySpinLock>::crange;

  size_t value;
};

struct B : public Registry<B> {};

void Manipulate() {
  for (size_t i = 0; i < 1000; ++i) {
    A a0;
    A a1{a0};
    A a2 = std::move(a0);
    a2 = a1;
    a2 = std::move(a1);
    A a3;
    // using std::swap; // TODO: MSVC ADL bug?
    swap(a2, a3);
  }
}

} // namespace

TEST(Registry, Base) {
  // Default constructor and destructor.
  EXPECT_EQ(A::size(), 0);
  {
    A a0;
    EXPECT_EQ(A::size(), 1);
  }

  EXPECT_EQ(A::size(), 0);
  {
    A a0;
    EXPECT_EQ(A::size(), 1);
    A a1;
    EXPECT_EQ(A::size(), 2);
  }
  EXPECT_EQ(A::size(), 0);

  {
    A a0;
    EXPECT_EQ(A::size(), 1);
    A a1;
    EXPECT_EQ(A::size(), 2);
    A a2;
    EXPECT_EQ(A::size(), 3);
  }
  EXPECT_EQ(A::size(), 0);

  // Move, copy and move constructors and operators.
  {
    A a0;
    EXPECT_EQ(A::size(), 1);

    A a1{a0};
    EXPECT_EQ(A::size(), 2);

    A a2 = std::move(a0);
    EXPECT_EQ(A::size(), 3);

    a2 = a1;
    EXPECT_EQ(A::size(), 3);

    a2 = std::move(a1);
    EXPECT_EQ(A::size(), 3);

    A a3;
    EXPECT_EQ(A::size(), 4);

    // using std::swap;
    swap(a2, a3);
    EXPECT_EQ(A::size(), 4);
  }
  EXPECT_EQ(A::size(), 0);
}

TEST(Registry, ThreadSafety) {
  {
    std::vector<std::jthread> threads(10);

    for (size_t i = 0; i < threads.size(); ++i) {
      auto &t = threads[i];
      t = std::jthread{Manipulate};
    }

    for (auto &t : threads) {
      t.join();
    }

    EXPECT_EQ(A::size(), 0);
  }
}

TEST(Registry, IteratorsAndRanges) {
  // Iterators.
  {
    A a0;
    A a1;
    A a2;
    A a3;
    A a4;
    a0.value = 3;
    a1.value = 6;
    a2.value = 9;
    a3.value = 12;
    a4.value = 15;

    size_t totalValue = 0;
    for (auto it = A::begin(); it != A::end(); ++it) {
      totalValue += it->value;
      totalValue += (*it).value;
      ++it->value;
    }

    for (auto it = A::cbegin(); it != A::cend(); ++it) {
      totalValue += it->value;
      totalValue += (*it).value;
      // ++it->value;
    }

    EXPECT_EQ(totalValue, 190);
  }

  // Ranges.
  {
    A a0;
    A a1;
    A a2;
    A a3;
    A a4;
    a0.value = 3;
    a1.value = 6;
    a2.value = 9;
    a3.value = 12;
    a4.value = 15;

    size_t totalValue = 0;
    for (auto &a : A::range()) {
      totalValue += a.value;
      ++a.value;
    }

    for (const auto &a : A::crange()) {
      totalValue += a.value;
      // ++a.value;
    }

    EXPECT_EQ(totalValue, 95);
  }

  // Ranges library.
  {
    A a0;
    A a1;
    A a2;
    A a3;
    A a4;
    a0.value = 3;
    a1.value = 6;
    a2.value = 9;
    a3.value = 12;
    a4.value = 15;

    size_t totalValue = 0;
    for (const auto &a : A::range() | std::views::take(3) | std::views::reverse) {
      totalValue += a.value;
    }

    for (const auto &a : A::crange() | std::views::take(3) | std::views::reverse) {
      totalValue += a.value;
    }

    EXPECT_EQ(totalValue, 36);
  }
}

} // namespace ARIA
