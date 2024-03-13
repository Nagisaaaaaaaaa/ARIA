#include "ARIA/BitVector.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(BitVector, Base) {
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
}

} // namespace ARIA
