#include "ARIA/BitVector.h"
#include "ARIA/Launcher.h"

#include <gtest/gtest.h>

#include <thread>

namespace ARIA {

namespace {

void TestThreadSafetyDevice() {
  size_t n = 10000;
  size_t nThreads = 100;

  // Fill.
  {
    BitVector<SpaceDevice, ThreadSafe> bitVector(n);

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        bitVector[i] = true;

    Launcher(nThreads, [=, span = bitVector.span()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span.Fill(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(bitVector[i], true);
  }

  {
    BitVector<SpaceDevice, ThreadSafe> bitVector(n);

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        bitVector[i] = true;

    Launcher(nThreads, [=, span = bitVector.rawSpan()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span.Fill(i);
      }

      // Test compile.
      for (auto it = span.begin(); it != ++span.begin(); ++it)
        ;
    }).Launch();

    cuda::device::current::get().synchronize();

    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(bitVector[i], true);
  }

  // Clear.
  {
    BitVector<SpaceDevice, ThreadSafe> bitVector(n);

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        bitVector[i] = true;

    Launcher(nThreads, [=, span = bitVector.span()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span.Clear(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(bitVector[i], false);
  }

  {
    BitVector<SpaceDevice, ThreadSafe> bitVector(n);

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        bitVector[i] = true;

    Launcher(nThreads, [=, span = bitVector.rawSpan()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span.Clear(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(bitVector[i], false);
  }

  // Flip.
  {
    BitVector<SpaceDevice, ThreadSafe> bitVector(n);

    Launcher(nThreads, [=, span = bitVector.span()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span.Flip(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(bitVector[i], true);
  }

  {
    BitVector<SpaceDevice, ThreadSafe> bitVector(n);

    Launcher(nThreads, [=, span = bitVector.rawSpan()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span.Flip(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(bitVector[i], true);
  }

  {
    BitVector<SpaceDevice, ThreadSafe> bitVector(n);

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        bitVector[i] = true;

    Launcher(nThreads, [=, span = bitVector.span()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span.Flip(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        EXPECT_EQ(bitVector[i], false);
      else
        EXPECT_EQ(bitVector[i], true);
  }

  {
    BitVector<SpaceDevice, ThreadSafe> bitVector(n);

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        bitVector[i] = true;

    Launcher(nThreads, [=, span = bitVector.rawSpan()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span.Flip(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        EXPECT_EQ(bitVector[i], false);
      else
        EXPECT_EQ(bitVector[i], true);
  }
}

} // namespace

TEST(BitVector, Base) {
  auto testBitVectorBase = []<typename TSpace, typename TThreadSafety>() {
    // Constructors.
    {
      BitVector<TSpace, TThreadSafety> bitVector;
      EXPECT_EQ(bitVector.size(), 0);
    }

    for (size_t n = 0; n < 100; ++n) {
      BitVector<TSpace, TThreadSafety> bitVector(n);
      EXPECT_EQ(bitVector.size(), n);

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector[i], false);
    }

    // Operator[].
    for (size_t n = 0; n < 100; ++n) {
      BitVector<TSpace, TThreadSafety> bitVector(n);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitVector[i] = true;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector[i], i % 2 == 0 ? true : false);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVector[i] = true;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector[i], true);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitVector[i] = false;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector[i], i % 2 == 0 ? false : true);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVector[i] = false;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector[i], false);
    }

    // Copy and move.
    {
      const size_t n = 10;
      BitVector<TSpace, TThreadSafety> bitVector0(n);
      for (size_t i = 0; i < n; ++i)
        bitVector0[i] = true;

      auto bitVector1 = bitVector0;
      EXPECT_EQ(bitVector0.size(), n);
      EXPECT_EQ(bitVector1.size(), n);
      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector0[i], true);
      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector1[i], true);

      auto bitVector2 = std::move(bitVector0);
      EXPECT_EQ(bitVector0.size(), 0);
      EXPECT_EQ(bitVector1.size(), n);
      EXPECT_EQ(bitVector2.size(), n);
      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector1[i], true);
      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector2[i], true);
    }

    // Resize.
    {
      const size_t n0 = 10;
      const size_t n1 = 30;

      BitVector<TSpace, TThreadSafety> bitVector(n0);
      for (size_t i = 0; i < n0; ++i)
        bitVector[i] = true;

      bitVector.resize(n1);
      EXPECT_EQ(bitVector.size(), n1);

      for (size_t i = 0; i < n1; ++i) {
        if (i < n0)
          EXPECT_EQ(bitVector[i], true);
        else
          EXPECT_EQ(bitVector[i], false);
      }

      for (size_t i = 0; i < n1; ++i)
        bitVector[i] = !bitVector[i];

      for (size_t i = 0; i < n1; ++i) {
        if (i < n0)
          EXPECT_EQ(bitVector[i], false);
        else
          EXPECT_EQ(bitVector[i], true);
      }
    }

    // Non-const `begin` and `end`.
    {
      const size_t n = 10;
      BitVector<TSpace, TThreadSafety> bitVector(n);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVector[i] = true;

      size_t i = 0;
      for (auto it = bitVector.begin(); it != bitVector.end(); ++it, ++i) {
        bool v = *it;

        if (i % 2 == 1) {
          EXPECT_TRUE(*it);
          EXPECT_TRUE(v);
          EXPECT_TRUE(it->value());
          EXPECT_EQ(*it, true);
          EXPECT_EQ(v, true);
          EXPECT_EQ(it->value(), true);
        } else {
          EXPECT_FALSE(*it);
          EXPECT_FALSE(v);
          EXPECT_FALSE(it->value());
          EXPECT_EQ(*it, false);
          EXPECT_EQ(v, false);
          EXPECT_EQ(it->value(), false);
        }
      }
      EXPECT_EQ(i, n);
    }

    // Const `begin` and `end`.
    {
      const size_t n = 10;
      BitVector<TSpace, TThreadSafety> bitVectorNonConst(n);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVectorNonConst[i] = true;

      const BitVector<TSpace, TThreadSafety> bitVector = bitVectorNonConst;

      size_t i = 0;
      for (auto it = bitVector.begin(); it != bitVector.end(); ++it, ++i) {
        bool v = *it;

        if (i % 2 == 1) {
          EXPECT_TRUE(*it);
          EXPECT_TRUE(v);
          EXPECT_TRUE(it->value());
          EXPECT_EQ(*it, true);
          EXPECT_EQ(v, true);
          EXPECT_EQ(it->value(), true);
        } else {
          EXPECT_FALSE(*it);
          EXPECT_FALSE(v);
          EXPECT_FALSE(it->value());
          EXPECT_EQ(*it, false);
          EXPECT_EQ(v, false);
          EXPECT_EQ(it->value(), false);
        }
      }
      EXPECT_EQ(i, n);
    }

    // `cbegin` and `cend`.
    {
      const size_t n = 10;
      BitVector<TSpace, TThreadSafety> bitVectorNonConst(n);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVectorNonConst[i] = true;

      const BitVector<TSpace, TThreadSafety> bitVector = bitVectorNonConst;

      size_t i = 0;
      for (auto it = bitVector.cbegin(); it != bitVector.cend(); ++it, ++i) {
        bool v = *it;

        if (i % 2 == 1) {
          EXPECT_TRUE(*it);
          EXPECT_TRUE(v);
          EXPECT_TRUE(it->value());
          EXPECT_EQ(*it, true);
          EXPECT_EQ(v, true);
          EXPECT_EQ(it->value(), true);
        } else {
          EXPECT_FALSE(*it);
          EXPECT_FALSE(v);
          EXPECT_FALSE(it->value());
          EXPECT_EQ(*it, false);
          EXPECT_EQ(v, false);
          EXPECT_EQ(it->value(), false);
        }
      }
      EXPECT_EQ(i, n);
    }

    // Non-const range-based for.
    {
      const size_t n = 10;
      BitVector<TSpace, TThreadSafety> bitVector(n);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVector[i] = true;

      size_t i = 0;
      for (Property auto v : bitVector) {
        if (i % 2 == 1) {
          EXPECT_TRUE(v);
          EXPECT_TRUE(v.value());
          EXPECT_EQ(v, true);
          EXPECT_EQ(v.value(), true);
        } else {
          EXPECT_FALSE(v);
          EXPECT_FALSE(v.value());
          EXPECT_EQ(v, false);
          EXPECT_EQ(v.value(), false);
        }
        ++i;
      }
      EXPECT_EQ(i, n);
    }

    // Const range-based for.
    {
      const size_t n = 10;
      BitVector<TSpace, TThreadSafety> bitVectorNonConst(n);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVectorNonConst[i] = true;

      const BitVector<TSpace, TThreadSafety> bitVector = bitVectorNonConst;

      size_t i = 0;
      for (Property auto v : bitVector) {
        if (i % 2 == 1) {
          EXPECT_TRUE(v);
          EXPECT_TRUE(v.value());
          EXPECT_EQ(v, true);
          EXPECT_EQ(v.value(), true);
        } else {
          EXPECT_FALSE(v);
          EXPECT_FALSE(v.value());
          EXPECT_EQ(v, false);
          EXPECT_EQ(v.value(), false);
        }
        ++i;
      }
      EXPECT_EQ(i, n);
    }

    // Iterator requirements.
    {
      BitVector<TSpace, TThreadSafety> t0;
      auto tSpan0 = t0.span();
      auto tRawSpan0 = t0.rawSpan();

      const BitVector<TSpace, TThreadSafety> t1;
      const auto tSpan1 = t1.span();
      const auto tRawSpan1 = t1.rawSpan();

      auto testIteratorRequirements = [](auto v) {
        static_assert(std::random_access_iterator<decltype(v.begin())>);
        static_assert(std::random_access_iterator<decltype(v.end())>);
        static_assert(std::random_access_iterator<decltype(v.cbegin())>);
        static_assert(std::random_access_iterator<decltype(v.cend())>);
      };

      testIteratorRequirements(t0);
      testIteratorRequirements(tSpan0);
      testIteratorRequirements(tRawSpan0);
      testIteratorRequirements(t1);
      testIteratorRequirements(tSpan1);
      testIteratorRequirements(tRawSpan1);
    }
  };

  testBitVectorBase.operator()<SpaceHost, ThreadUnsafe>();
  testBitVectorBase.operator()<SpaceHost, ThreadSafe>();
  testBitVectorBase.operator()<SpaceDevice, ThreadUnsafe>();
  testBitVectorBase.operator()<SpaceDevice, ThreadSafe>();
}

TEST(BitVector, Span) {
  auto testBitVectorSpan = []<typename TSpace, typename TThreadSafety>() {
    // Constructors.
    {
      BitVector<SpaceHost, ThreadSafe> bitVector;
      BitVectorSpan bitVectorSpan = bitVector.span();
      EXPECT_EQ(bitVectorSpan.size(), 0);
    }

    for (size_t n = 0; n < 100; ++n) {
      BitVector<SpaceHost, ThreadSafe> bitVector(n);
      BitVectorSpan bitVectorSpan = bitVector.span();
      EXPECT_EQ(bitVectorSpan.size(), n);

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVectorSpan[i], false);
    }

    // Operator[].
    for (size_t n = 0; n < 100; ++n) {
      BitVector<SpaceHost, ThreadSafe> bitVector(n);
      BitVectorSpan bitVectorSpan = bitVector.span();

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitVectorSpan[i] = true;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVectorSpan[i], i % 2 == 0 ? true : false);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVectorSpan[i] = true;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVectorSpan[i], true);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitVectorSpan[i] = false;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVectorSpan[i], i % 2 == 0 ? false : true);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVectorSpan[i] = false;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVectorSpan[i], false);
    }
  };

  auto testBitVectorRawSpan = []<typename TSpace, typename TThreadSafety>() {
    // Constructors.
    {
      BitVector<SpaceHost, ThreadSafe> bitVector;
      BitVectorSpan bitVectorSpan = bitVector.rawSpan();
      EXPECT_EQ(bitVectorSpan.size(), 0);
    }

    for (size_t n = 0; n < 100; ++n) {
      BitVector<SpaceHost, ThreadSafe> bitVector(n);
      BitVectorSpan bitVectorSpan = bitVector.rawSpan();
      EXPECT_EQ(bitVectorSpan.size(), n);

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVectorSpan[i], false);
    }

    // Operator[].
    for (size_t n = 0; n < 100; ++n) {
      BitVector<SpaceHost, ThreadSafe> bitVector(n);
      BitVectorSpan bitVectorSpan = bitVector.rawSpan();

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitVectorSpan[i] = true;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVectorSpan[i], i % 2 == 0 ? true : false);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVectorSpan[i] = true;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVectorSpan[i], true);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitVectorSpan[i] = false;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVectorSpan[i], i % 2 == 0 ? false : true);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVectorSpan[i] = false;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVectorSpan[i], false);
    }
  };

  testBitVectorSpan.operator()<SpaceHost, ThreadUnsafe>();
  testBitVectorSpan.operator()<SpaceHost, ThreadSafe>();
  testBitVectorSpan.operator()<SpaceDevice, ThreadUnsafe>();
  testBitVectorSpan.operator()<SpaceDevice, ThreadSafe>();

  testBitVectorRawSpan.operator()<SpaceHost, ThreadUnsafe>();
  testBitVectorRawSpan.operator()<SpaceHost, ThreadSafe>();
}

TEST(BitVector, ThreadSafetyHost) {
  auto getRef = [](BitVector<SpaceHost, ThreadSafe> &v) -> BitVector<SpaceHost, ThreadSafe> & { return v; };
  auto getSpan = [](BitVector<SpaceHost, ThreadSafe> &v) { return v.span(); };
  auto getRawSpan = [](BitVector<SpaceHost, ThreadSafe> &v) { return v.rawSpan(); };

  auto testByGetter = [](const auto &getter) {
    size_t n = 100000;
    size_t nThreads = 10;

    // Fill.
    {
      BitVector<SpaceHost, ThreadSafe> bitVector(n);
      std::vector<std::jthread> threads(nThreads);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitVector[i] = true;

      for (size_t t = 0; t < nThreads; ++t)
        threads[t] = std::jthread{[n, nThreads, &bitVector, t, &getter]() mutable {
          size_t tCpy = t;
          for (size_t i = tCpy; i < n; i += nThreads) {
            getter(bitVector).Fill(i);
          }
        }};

      for (auto &t : threads)
        t.join();

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector[i], true);
    }

    // Clear.
    {
      BitVector<SpaceHost, ThreadSafe> bitVector(n);
      std::vector<std::jthread> threads(nThreads);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitVector[i] = true;

      for (size_t t = 0; t < nThreads; ++t)
        threads[t] = std::jthread{[n, nThreads, &bitVector, t, &getter]() mutable {
          size_t tCpy = t;
          for (size_t i = tCpy; i < n; i += nThreads) {
            getter(bitVector).Clear(i);
          }
        }};

      for (auto &t : threads)
        t.join();

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector[i], false);
    }

    // Flip.
    {
      BitVector<SpaceHost, ThreadSafe> bitVector(n);
      std::vector<std::jthread> threads(nThreads);

      for (size_t t = 0; t < nThreads; ++t)
        threads[t] = std::jthread{[n, nThreads, &bitVector, t, &getter]() mutable {
          size_t tCpy = t;
          for (size_t i = tCpy; i < n; i += nThreads) {
            getter(bitVector).Flip(i);
          }
        }};

      for (auto &t : threads)
        t.join();

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector[i], true);
    }

    {
      BitVector<SpaceHost, ThreadSafe> bitVector(n);
      std::vector<std::jthread> threads(nThreads);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitVector[i] = true;

      for (size_t t = 0; t < nThreads; ++t)
        threads[t] = std::jthread{[n, nThreads, &bitVector, t, &getter]() mutable {
          size_t tCpy = t;
          for (size_t i = tCpy; i < n; i += nThreads) {
            getter(bitVector).Flip(i);
          }
        }};

      for (auto &t : threads)
        t.join();

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          EXPECT_EQ(bitVector[i], false);
        else
          EXPECT_EQ(bitVector[i], true);
    }
  };

  testByGetter(getRef);
  testByGetter(getSpan);
  testByGetter(getRawSpan);
}

TEST(BitVector, ThreadSafetyDevice) {
  TestThreadSafetyDevice();
}

} // namespace ARIA
