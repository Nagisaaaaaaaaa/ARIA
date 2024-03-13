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
}

} // namespace ARIA
