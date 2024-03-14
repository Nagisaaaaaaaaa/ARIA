#include "ARIA/BitVector.h"

#include <gtest/gtest.h>

#include <thread>

namespace ARIA {

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

TEST(BitVector, ThreadSafety) {
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
      threads[t] = std::jthread{[n, nThreads, &bitVector, t]() {
        size_t tCpy = t;
        for (size_t i = tCpy; i < n; i += nThreads) {
          bitVector.Fill(i);
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
      threads[t] = std::jthread{[n, nThreads, &bitVector, t]() {
        size_t tCpy = t;
        for (size_t i = tCpy; i < n; i += nThreads) {
          bitVector.Clear(i);
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
      threads[t] = std::jthread{[n, nThreads, &bitVector, t]() {
        size_t tCpy = t;
        for (size_t i = tCpy; i < n; i += nThreads) {
          bitVector.Flip(i);
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
      threads[t] = std::jthread{[n, nThreads, &bitVector, t]() {
        size_t tCpy = t;
        for (size_t i = tCpy; i < n; i += nThreads) {
          bitVector.Flip(i);
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
}

} // namespace ARIA
