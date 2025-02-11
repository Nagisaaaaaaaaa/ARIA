#include "ARIA/MarchingCube.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(MarchingCube, Base) {
  auto computeT = [](Real v0, Real v1, Real isoValue) { return (isoValue - v0) / (v1 - v0); };

  // 1D.
  {
    using MC = MarchingCube<1>;
    Vec1r positions[2] = {Vec1r{-2.5_R}, Vec1r{2.5_R}};
    Real values[2] = {0.1_R, 0.8_R};
    Real isoValue = 0.4_R;
    MC::Extract([&](uint i) { return positions[i]; }, [&](uint i) { return values[i]; }, isoValue,
                [&](cuda::std::array<Vec1r, 1> primitiveVertices) {
      Vec1r p = Lerp(positions[0], positions[1], computeT(values[0], values[1], isoValue));
      EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p.x());
    });
  }
}

} // namespace ARIA
