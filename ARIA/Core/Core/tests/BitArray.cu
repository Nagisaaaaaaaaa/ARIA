#include "ARIA/BitArray.h"
#include "ARIA/ForEach.h"
#include "ARIA/Launcher.h"

#include <gtest/gtest.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace ARIA {

namespace {

void TestThreadSafetyDevice() {
  constexpr size_t n = 10000;
  size_t nThreads = 100;

  // Fill.
  {
    thrust::host_vector<BitArray<n, ThreadSafe>> bitArraysH(1);
    thrust::device_vector<BitArray<n, ThreadSafe>> bitArraysD(1);

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        bitArraysH[0][i] = true;

    bitArraysD = bitArraysH;
    Launcher(nThreads, [=, span = bitArraysD.data()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span->Fill(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    bitArraysH = bitArraysD;
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(bitArraysH[0][i], true);
  }

  // Clear.
  {
    thrust::host_vector<BitArray<n, ThreadSafe>> bitArraysH(1);
    thrust::device_vector<BitArray<n, ThreadSafe>> bitArraysD(1);

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        bitArraysH[0][i] = true;

    bitArraysD = bitArraysH;
    Launcher(nThreads, [=, span = bitArraysD.data()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span->Clear(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    bitArraysH = bitArraysD;
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(bitArraysH[0][i], false);
  }

  // Flip.
  {
    thrust::host_vector<BitArray<n, ThreadSafe>> bitArraysH(1);
    thrust::device_vector<BitArray<n, ThreadSafe>> bitArraysD(1);

    bitArraysD = bitArraysH;
    Launcher(nThreads, [=, span = bitArraysD.data()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span->Flip(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    bitArraysH = bitArraysD;
    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(bitArraysH[0][i], true);
  }

  {
    thrust::host_vector<BitArray<n, ThreadSafe>> bitArraysH(1);
    thrust::device_vector<BitArray<n, ThreadSafe>> bitArraysD(1);

    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        bitArraysH[0][i] = true;

    bitArraysD = bitArraysH;
    Launcher(nThreads, [=, span = bitArraysD.data()] ARIA_DEVICE(size_t t) mutable {
      size_t tCpy = t;
      for (size_t i = tCpy; i < n; i += nThreads) {
        span->Flip(i);
      }
    }).Launch();

    cuda::device::current::get().synchronize();

    bitArraysH = bitArraysD;
    for (size_t i = 0; i < n; ++i)
      if (i % 2 == 0)
        EXPECT_EQ(bitArraysH[0][i], false);
      else
        EXPECT_EQ(bitArraysH[0][i], true);
  }
}

} // namespace

TEST(BitArray, Base) {
  auto testBitArrayBase = []<typename TThreadSafety>() {
    // Constructors.
    {
      BitArray<0, TThreadSafety> bitArray;
      static_assert(bitArray.size() == 0);
    }

    ForEach<100>([]<auto n>() {
      BitArray<n, TThreadSafety> bitArray;
      static_assert(bitArray.size() == n);

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitArray[i], false);
    });

    // Operator[].
    ForEach<100>([]<auto n>() {
      BitArray<n, TThreadSafety> bitArray;

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitArray[i] = true;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitArray[i], i % 2 == 0 ? true : false);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitArray[i] = true;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitArray[i], true);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitArray[i] = false;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitArray[i], i % 2 == 0 ? false : true);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitArray[i] = false;

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitArray[i], false);
    });

    // Copy.
    constexpr size_t n = 10;
    {
      BitArray<n, TThreadSafety> bitArray0;
      for (size_t i = 0; i < n; ++i)
        bitArray0[i] = true;

      auto bitArray1 = bitArray0;
      EXPECT_EQ(bitArray0.size(), n);
      EXPECT_EQ(bitArray1.size(), n);
      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitArray0[i], true);
      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitArray1[i], true);
    }

    // Non-const `begin` and `end`.
    {
      BitArray<n, TThreadSafety> bitArray;

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitArray[i] = true;

      size_t i = 0;
      for (auto it = bitArray.begin(); it != bitArray.end(); ++it, ++i) {
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
      BitArray<n, TThreadSafety> bitArrayNonConst;

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitArrayNonConst[i] = true;

      const BitArray<n, TThreadSafety> bitArray = bitArrayNonConst;

      size_t i = 0;
      for (auto it = bitArray.begin(); it != bitArray.end(); ++it, ++i) {
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
      BitArray<n, TThreadSafety> bitArrayNonConst;

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitArrayNonConst[i] = true;

      const BitArray<n, TThreadSafety> bitArray = bitArrayNonConst;

      size_t i = 0;
      for (auto it = bitArray.cbegin(); it != bitArray.cend(); ++it, ++i) {
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
      BitArray<n, TThreadSafety> bitArray;

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitArray[i] = true;

      size_t i = 0;
      for (Property auto v : bitArray) {
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
      BitArray<n, TThreadSafety> bitArrayNonConst;

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitArrayNonConst[i] = true;

      const BitArray<n, TThreadSafety> bitArray = bitArrayNonConst;

      size_t i = 0;
      for (Property auto v : bitArray) {
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
      BitArray<10, TThreadSafety> t0;
      const BitArray<10, TThreadSafety> t1;

      auto testIteratorRequirements = [](auto v) {
        static_assert(std::random_access_iterator<decltype(v.begin())>);
        static_assert(std::random_access_iterator<decltype(v.end())>);
        static_assert(std::random_access_iterator<decltype(v.cbegin())>);
        static_assert(std::random_access_iterator<decltype(v.cend())>);
      };

      testIteratorRequirements(t0);
      testIteratorRequirements(t1);
    }
  };

  testBitArrayBase.operator()<ThreadUnsafe>();
  testBitArrayBase.operator()<ThreadSafe>();
}

TEST(BitArray, ThreadSafetyHost) {
  constexpr size_t n = 100000;

  auto getRef = [](BitArray<n, ThreadSafe> &v) -> BitArray<n, ThreadSafe> & { return v; };

  auto testByGetter = [](const auto &getter) {
    size_t nThreads = 10;

    // Fill.
    {
      BitArray<n, ThreadSafe> bitArray;
      std::vector<std::jthread> threads(nThreads);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitArray[i] = true;

      for (size_t t = 0; t < nThreads; ++t)
        threads[t] = std::jthread{[nThreads, &bitArray, t, &getter]() mutable {
          size_t tCpy = t;
          for (size_t i = tCpy; i < n; i += nThreads) {
            getter(bitArray).Fill(i);
          }
        }};

      for (auto &t : threads)
        t.join();

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitArray[i], true);
    }

    // Clear.
    {
      BitArray<n, ThreadSafe> bitArray;
      std::vector<std::jthread> threads(nThreads);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitArray[i] = true;

      for (size_t t = 0; t < nThreads; ++t)
        threads[t] = std::jthread{[nThreads, &bitArray, t, &getter]() mutable {
          size_t tCpy = t;
          for (size_t i = tCpy; i < n; i += nThreads) {
            getter(bitArray).Clear(i);
          }
        }};

      for (auto &t : threads)
        t.join();

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitArray[i], false);
    }

    // Flip.
    {
      BitArray<n, ThreadSafe> bitArray;
      std::vector<std::jthread> threads(nThreads);

      for (size_t t = 0; t < nThreads; ++t)
        threads[t] = std::jthread{[nThreads, &bitArray, t, &getter]() mutable {
          size_t tCpy = t;
          for (size_t i = tCpy; i < n; i += nThreads) {
            getter(bitArray).Flip(i);
          }
        }};

      for (auto &t : threads)
        t.join();

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitArray[i], true);
    }

    {
      BitArray<n, ThreadSafe> bitArray;
      std::vector<std::jthread> threads(nThreads);

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          bitArray[i] = true;

      for (size_t t = 0; t < nThreads; ++t)
        threads[t] = std::jthread{[nThreads, &bitArray, t, &getter]() mutable {
          size_t tCpy = t;
          for (size_t i = tCpy; i < n; i += nThreads) {
            getter(bitArray).Flip(i);
          }
        }};

      for (auto &t : threads)
        t.join();

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 0)
          EXPECT_EQ(bitArray[i], false);
        else
          EXPECT_EQ(bitArray[i], true);
    }
  };

  testByGetter(getRef);
}

TEST(BitArray, ThreadSafetyDevice) {
  TestThreadSafetyDevice();
}

} // namespace ARIA
