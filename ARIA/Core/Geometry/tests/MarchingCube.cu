#include "ARIA/MarchingCube.h"

#include <gtest/gtest.h>

#include <cuda/std/span>

namespace ARIA {

TEST(MarchingCube, Base) {
  using arr = std::array;
  auto computeT = [](Real v0, Real v1, Real isoValue) { return (isoValue - v0) / (v1 - v0); };

  // 1D.
  {
    using MC = MarchingCube<1>;

    auto testExtract_AA = [&](const auto &positions, const auto &values, Real isoValue) {
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec1r, 1> &) { EXPECT_FALSE(true); });
    };

    auto testExtract_AB = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec1r, 1> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        Vec1r p = Lerp(positions(0), positions(1), computeT(values(0), values(1), isoValue));
        EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p.x());
      });
      EXPECT_EQ(times, 1);
    };

    auto positions = [](uint i) { return arr{Vec1r{-2.5_R}, Vec1r{2.5_R}}[i]; };
    auto valuesOO = [](uint i) { return arr{0.1_R, 0.1_R}[i]; };
    auto valuesPO = [](uint i) { return arr{0.8_R, 0.1_R}[i]; };
    auto valuesOP = [](uint i) { return arr{0.1_R, 0.8_R}[i]; };
    auto valuesPP = [](uint i) { return arr{0.8_R, 0.8_R}[i]; };
    Real isoValue = 0.4_R;

    testExtract_AA(positions, valuesOO, isoValue);
    testExtract_AA(positions, valuesPP, isoValue);
    testExtract_AB(positions, valuesPO, isoValue);
    testExtract_AB(positions, valuesOP, isoValue);
  }

  // 2D.
  {
    using MC = MarchingCube<2>;

    auto testExtract_AAAA = [&](const auto &positions, const auto &values, Real isoValue) {
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &) { EXPECT_FALSE(true); });
    };

    auto testExtract_ABBB = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        std::ranges::sort(primitiveVertices, [](const Vec2r &a, const Vec2r &b) {
          return a.y() < b.y() || (a.y() == b.y() && a.x() < b.x());
        });
        Vec2r p_00_10 = Lerp(positions(0, 0), positions(1, 0), computeT(values(0, 0), values(1, 0), isoValue));
        Vec2r p_00_01 = Lerp(positions(0, 0), positions(0, 1), computeT(values(0, 0), values(0, 1), isoValue));
        EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p_00_10.x());
        EXPECT_FLOAT_EQ(primitiveVertices[0].y(), p_00_10.y());
        EXPECT_FLOAT_EQ(primitiveVertices[1].x(), p_00_01.x());
        EXPECT_FLOAT_EQ(primitiveVertices[1].y(), p_00_01.y());
      });
      EXPECT_EQ(times, 1);
    };

    auto positions = [](uint i, uint j) {
      return arr{arr{Vec2r{-2.5_R, -2.5_R}, Vec2r{-2.5_R, 2.5_R}}, //
                 arr{Vec2r{2.5_R, -2.5_R}, Vec2r{2.5_R, 2.5_R}}}[i][j];
    };

    auto values_OOOO = [](uint i, uint j) { return arr{arr{0.1_R, 0.1_R}, arr{0.1_R, 0.1_R}}[i][j]; };
    auto values_PPPP = [](uint i, uint j) { return arr{arr{0.8_R, 0.8_R}, arr{0.8_R, 0.8_R}}[i][j]; };

    auto values_POOO = [](uint i, uint j) { return arr{arr{0.8_R, 0.1_R}, arr{0.1_R, 0.1_R}}[i][j]; };
    auto values_OPPP = [](uint i, uint j) { return arr{arr{0.1_R, 0.8_R}, arr{0.8_R, 0.8_R}}[i][j]; };

    auto values_OPOO = [](uint i, uint j) { return arr{arr{0.1_R, 0.8_R}, arr{0.1_R, 0.1_R}}[i][j]; };
    auto values_POPP = [](uint i, uint j) { return arr{arr{0.8_R, 0.1_R}, arr{0.8_R, 0.8_R}}[i][j]; };

    Real isoValue = 0.4_R;

    testExtract_AAAA(positions, values_OOOO, isoValue);
    testExtract_AAAA(positions, values_PPPP, isoValue);

    testExtract_ABBB(positions, values_POOO, isoValue);
    testExtract_ABBB(positions, values_OPPP, isoValue);
  }
}

} // namespace ARIA
