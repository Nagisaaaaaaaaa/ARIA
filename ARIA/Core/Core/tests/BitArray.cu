#include "ARIA/BitArray.h"
#include "ARIA/ForEach.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(BitArray, Base) {
  auto testBitArrayBase = []<typename TThreadSafety>() {
    // Constructors.
    {
      BitArray<0, TThreadSafety> bitVector;
      static_assert(bitVector.size() == 0);
    }

    ForEach<100>([]<auto n>() {
      BitArray<n, TThreadSafety> bitVector;
      static_assert(bitVector.size() == n);

      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector[i], false);
    });

    // Operator[].
    ForEach<100>([]<auto n>() {
      BitArray<n, TThreadSafety> bitVector;

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
    });

    // Copy.
    {
      constexpr size_t n = 10;
      BitArray<n, TThreadSafety> bitVector0;
      for (size_t i = 0; i < n; ++i)
        bitVector0[i] = true;

      auto bitVector1 = bitVector0;
      EXPECT_EQ(bitVector0.size(), n);
      EXPECT_EQ(bitVector1.size(), n);
      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector0[i], true);
      for (size_t i = 0; i < n; ++i)
        EXPECT_EQ(bitVector1[i], true);
    }

    // Non-const `begin` and `end`.
    {
      constexpr size_t n = 10;
      BitArray<n, TThreadSafety> bitVector;

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
      constexpr size_t n = 10;
      BitArray<n, TThreadSafety> bitVectorNonConst;

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVectorNonConst[i] = true;

      const BitArray<n, TThreadSafety> bitVector = bitVectorNonConst;

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
      constexpr size_t n = 10;
      BitArray<n, TThreadSafety> bitVectorNonConst;

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVectorNonConst[i] = true;

      const BitArray<n, TThreadSafety> bitVector = bitVectorNonConst;

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
      constexpr size_t n = 10;
      BitArray<n, TThreadSafety> bitVector;

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
      constexpr size_t n = 10;
      BitArray<n, TThreadSafety> bitVectorNonConst;

      for (size_t i = 0; i < n; ++i)
        if (i % 2 == 1)
          bitVectorNonConst[i] = true;

      const BitArray<n, TThreadSafety> bitVector = bitVectorNonConst;

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

} // namespace ARIA
