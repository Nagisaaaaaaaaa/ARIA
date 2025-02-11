#include "ARIA/MarchingCube.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(MarchingCube, Base) {
  auto computeT = [](Real v0, Real v1, Real isoValue) { return (isoValue - v0) / (v1 - v0); };

  // 1D.
  {
    using MC = MarchingCube<1>;

    auto testExtract0 = [&](const auto &positions, const auto &values, Real isoValue) {
      MC::Extract([&](uint i) { return positions[i]; }, [&](uint i) { return values[i]; }, isoValue,
                  [&](cuda::std::array<Vec1r, 1>) { EXPECT_FALSE(true); });
    };

    auto testExtract1 = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract([&](uint i) { return positions[i]; }, [&](uint i) { return values[i]; }, isoValue,
                  [&](cuda::std::array<Vec1r, 1> primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        Vec1r p = Lerp(positions[0], positions[1], computeT(values[0], values[1], isoValue));
        EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p.x());
      });
      EXPECT_EQ(times, 1);
    };

    std::array positions = {Vec1r{-2.5_R}, Vec1r{2.5_R}};
    std::array values0 = {0.1_R, 0.1_R};
    std::array values1 = {0.8_R, 0.1_R};
    std::array values2 = {0.1_R, 0.8_R};
    std::array values3 = {0.8_R, 0.8_R};
    Real isoValue = 0.4_R;

    testExtract0(positions, values0, isoValue);
    testExtract1(positions, values1, isoValue);
    testExtract1(positions, values2, isoValue);
    testExtract0(positions, values3, isoValue);
  }
}

} // namespace ARIA
