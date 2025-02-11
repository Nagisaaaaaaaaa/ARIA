#include "ARIA/MarchingCube.h"

#include <gtest/gtest.h>

#include <cuda/std/span>

namespace ARIA {

TEST(MarchingCube, Base) {
  auto computeT = [](Real v0, Real v1, Real isoValue) { return (isoValue - v0) / (v1 - v0); };

  // 1D.
  {
    using MC = MarchingCube<1>;

    auto testExtract0 = [&](const auto &positions, const auto &values, Real isoValue) {
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec1r, 1> &) { EXPECT_FALSE(true); });
    };

    auto testExtract1 = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec1r, 1> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        Vec1r p = Lerp(positions(0), positions(1), computeT(values(0), values(1), isoValue));
        EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p.x());
      });
      EXPECT_EQ(times, 1);
    };

    auto positions = [](uint i) { return std::array{Vec1r{-2.5_R}, Vec1r{2.5_R}}[i]; };
    auto valuesOO = [](uint i) { return std::array{0.1_R, 0.1_R}[i]; };
    auto valuesPO = [](uint i) { return std::array{0.8_R, 0.1_R}[i]; };
    auto valuesOP = [](uint i) { return std::array{0.1_R, 0.8_R}[i]; };
    auto valuesPP = [](uint i) { return std::array{0.8_R, 0.8_R}[i]; };
    Real isoValue = 0.4_R;

    testExtract0(positions, valuesOO, isoValue);
    testExtract1(positions, valuesPO, isoValue);
    testExtract1(positions, valuesOP, isoValue);
    testExtract0(positions, valuesPP, isoValue);
  }

  // 2D.
  {
    using MC = MarchingCube<2>;

    auto testExtract0 = [&](const auto &positions, const auto &values, Real isoValue) {
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &) { EXPECT_FALSE(true); });
    };

    auto testExtract1_POOO = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        Vec2r p_00_10 = Lerp(positions(0, 0), positions(1, 0), computeT(values(0, 0), values(1, 0), isoValue));
        Vec2r p_00_01 = Lerp(positions(0, 0), positions(0, 1), computeT(values(0, 0), values(0, 1), isoValue));
      });
      EXPECT_EQ(times, 1);
    };

    auto positions = [](uint i, uint j) {
      return std::array{std::array{Vec2r{-2.5_R, -2.5_R}, Vec2r{-2.5_R, 2.5_R}}, //
                        std::array{Vec2r{2.5_R, -2.5_R}, Vec2r{2.5_R, 2.5_R}}}[i][j];
    };
    auto values_OOOO = [](uint i, uint j) {
      return std::array{std::array{0.1_R, 0.1_R}, //
                        std::array{0.1_R, 0.1_R}}[i][j];
    };
    auto values_POOO = [](uint i, uint j) {
      return std::array{std::array{0.8_R, 0.1_R}, //
                        std::array{0.1_R, 0.1_R}}[i][j];
    };
    Real isoValue = 0.4_R;

    testExtract0(positions, values_OOOO, isoValue);
    testExtract1_POOO(positions, values_POOO, isoValue);
  }
}

} // namespace ARIA
