#include "ARIA/BitVector.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(BitVector, Base) {
  // Constructors.
  {
    BitVector<SpaceHost, ThreadSafe> bitVector;
    EXPECT_EQ(bitVector.size(), 0);
  }

  for (size_t n = 0; n < 100; ++n) {
    BitVector<SpaceHost, ThreadSafe> bitVector(n);
    EXPECT_EQ(bitVector.size(), n);

    for (size_t i = 0; i < n; ++i)
      EXPECT_EQ(bitVector[i], false);
  }

  // Operator[].
  for (size_t n = 0; n < 100; ++n) {
    BitVector<SpaceHost, ThreadSafe> bitVector(n);

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
    BitVector<SpaceHost, ThreadSafe> bitVector0(n);
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

    BitVector<SpaceHost, ThreadSafe> bitVector(n0);
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
}

} // namespace ARIA