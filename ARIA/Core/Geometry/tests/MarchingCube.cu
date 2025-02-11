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
    auto values0 = [](uint i) { return std::array{0.1_R, 0.1_R}[i]; };
    auto values1 = [](uint i) { return std::array{0.8_R, 0.1_R}[i]; };
    auto values2 = [](uint i) { return std::array{0.1_R, 0.8_R}[i]; };
    auto values3 = [](uint i) { return std::array{0.8_R, 0.8_R}[i]; };
    Real isoValue = 0.4_R;

    testExtract0(positions, values0, isoValue);
    testExtract1(positions, values1, isoValue);
    testExtract1(positions, values2, isoValue);
    testExtract0(positions, values3, isoValue);
  }

  // 2D.
  {
    using MC = MarchingCube<2>;

    auto testExtract0 = [&](const auto &positions, const auto &values, Real isoValue) {
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &) { EXPECT_FALSE(true); });
    };

    auto testExtract1_OO = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        Vec2r p_OO_PO = Lerp(positions(0, 0), positions(1, 0), computeT(values(0, 0), values(1, 0), isoValue));
        Vec2r p_OO_OP = Lerp(positions(0, 0), positions(0, 1), computeT(values(0, 0), values(0, 1), isoValue));
        fmt::print("{} {} {} {}", p_OO_PO.x(), p_OO_PO.y(), p_OO_OP.x(), p_OO_OP.y());
        fmt::print("{} {} {} {}", primitiveVertices[0].x(), primitiveVertices[0].y(), primitiveVertices[1].x(),
                   primitiveVertices[1].y());
      });
      EXPECT_EQ(times, 1);
    };

    auto positions = [](uint i, uint j) {
      return std::array{std::array{Vec2r{-2.5_R, -2.5_R}, Vec2r{-2.5_R, 2.5_R}}, //
                        std::array{Vec2r{2.5_R, -2.5_R}, Vec2r{2.5_R, 2.5_R}}}[i][j];
    };
    auto values0 = [](uint i, uint j) {
      return std::array{std::array{0.1_R, 0.1_R}, //
                        std::array{0.1_R, 0.1_R}}[i][j];
    };
    Real isoValue = 0.4_R;

    testExtract0(positions, values0, isoValue);
  }
}

} // namespace ARIA
