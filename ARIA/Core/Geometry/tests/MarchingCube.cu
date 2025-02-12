#include "ARIA/MarchingCube.h"

#include <gtest/gtest.h>

#include <cuda/std/span>

namespace ARIA {

namespace {

template <typename T, size_t n>
using arr = std::array<T, n>;

}

TEST(MarchingCube, Base) {
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

    auto sortPrimitiveVertices = [](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
      std::ranges::sort(primitiveVertices, [](const Vec2r &a, const Vec2r &b) {
        return a.y() < b.y() || (a.y() == b.y() && a.x() < b.x());
      });
    };

    auto testExtract_AAAA = [&](const auto &positions, const auto &values, Real isoValue) {
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &) { EXPECT_FALSE(true); });
    };

    auto testExtract_BAAA = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        sortPrimitiveVertices(primitiveVertices);
        Vec2r p_00_10 = Lerp(positions(0, 0), positions(1, 0), computeT(values(0, 0), values(1, 0), isoValue));
        Vec2r p_00_01 = Lerp(positions(0, 0), positions(0, 1), computeT(values(0, 0), values(0, 1), isoValue));
        EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p_00_10.x());
        EXPECT_FLOAT_EQ(primitiveVertices[0].y(), p_00_10.y());
        EXPECT_FLOAT_EQ(primitiveVertices[1].x(), p_00_01.x());
        EXPECT_FLOAT_EQ(primitiveVertices[1].y(), p_00_01.y());
      });
      EXPECT_EQ(times, 1);
    };

    auto testExtract_ABAA = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        sortPrimitiveVertices(primitiveVertices);
        Vec2r p_00_01 = Lerp(positions(0, 0), positions(0, 1), computeT(values(0, 0), values(0, 1), isoValue));
        Vec2r p_01_11 = Lerp(positions(0, 1), positions(1, 1), computeT(values(0, 1), values(1, 1), isoValue));
        EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p_00_01.x());
        EXPECT_FLOAT_EQ(primitiveVertices[0].y(), p_00_01.y());
        EXPECT_FLOAT_EQ(primitiveVertices[1].x(), p_01_11.x());
        EXPECT_FLOAT_EQ(primitiveVertices[1].y(), p_01_11.y());
      });
      EXPECT_EQ(times, 1);
    };

    auto testExtract_AABA = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        sortPrimitiveVertices(primitiveVertices);
        Vec2r p_00_10 = Lerp(positions(0, 0), positions(1, 0), computeT(values(0, 0), values(1, 0), isoValue));
        Vec2r p_10_11 = Lerp(positions(1, 0), positions(1, 1), computeT(values(1, 0), values(1, 1), isoValue));
        EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p_00_10.x());
        EXPECT_FLOAT_EQ(primitiveVertices[0].y(), p_00_10.y());
        EXPECT_FLOAT_EQ(primitiveVertices[1].x(), p_10_11.x());
        EXPECT_FLOAT_EQ(primitiveVertices[1].y(), p_10_11.y());
      });
      EXPECT_EQ(times, 1);
    };

    auto testExtract_AAAB = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        sortPrimitiveVertices(primitiveVertices);
        Vec2r p_10_11 = Lerp(positions(1, 0), positions(1, 1), computeT(values(1, 0), values(1, 1), isoValue));
        Vec2r p_01_11 = Lerp(positions(0, 1), positions(1, 1), computeT(values(0, 1), values(1, 1), isoValue));
        EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p_10_11.x());
        EXPECT_FLOAT_EQ(primitiveVertices[0].y(), p_10_11.y());
        EXPECT_FLOAT_EQ(primitiveVertices[1].x(), p_01_11.x());
        EXPECT_FLOAT_EQ(primitiveVertices[1].y(), p_01_11.y());
      });
      EXPECT_EQ(times, 1);
    };

    auto testExtract_BBAA = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        sortPrimitiveVertices(primitiveVertices);
        Vec2r p_00_10 = Lerp(positions(0, 0), positions(1, 0), computeT(values(0, 0), values(1, 0), isoValue));
        Vec2r p_01_11 = Lerp(positions(0, 1), positions(1, 1), computeT(values(0, 1), values(1, 1), isoValue));
        EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p_00_10.x());
        EXPECT_FLOAT_EQ(primitiveVertices[0].y(), p_00_10.y());
        EXPECT_FLOAT_EQ(primitiveVertices[1].x(), p_01_11.x());
        EXPECT_FLOAT_EQ(primitiveVertices[1].y(), p_01_11.y());
      });
      EXPECT_EQ(times, 1);
    };

    auto testExtract_BABA = [&](const auto &positions, const auto &values, Real isoValue) {
      uint times = 0;
      MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
        EXPECT_EQ(times, 0);
        ++times;
        sortPrimitiveVertices(primitiveVertices);
        Vec2r p_00_01 = Lerp(positions(0, 0), positions(0, 1), computeT(values(0, 0), values(0, 1), isoValue));
        Vec2r p_10_11 = Lerp(positions(1, 0), positions(1, 1), computeT(values(1, 0), values(1, 1), isoValue));
        EXPECT_FLOAT_EQ(primitiveVertices[0].x(), p_00_01.x());
        EXPECT_FLOAT_EQ(primitiveVertices[0].y(), p_00_01.y());
        EXPECT_FLOAT_EQ(primitiveVertices[1].x(), p_10_11.x());
        EXPECT_FLOAT_EQ(primitiveVertices[1].y(), p_10_11.y());
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

    auto values_OOPO = [](uint i, uint j) { return arr{arr{0.1_R, 0.1_R}, arr{0.8_R, 0.1_R}}[i][j]; };
    auto values_PPOP = [](uint i, uint j) { return arr{arr{0.8_R, 0.8_R}, arr{0.1_R, 0.8_R}}[i][j]; };

    auto values_OOOP = [](uint i, uint j) { return arr{arr{0.1_R, 0.1_R}, arr{0.1_R, 0.8_R}}[i][j]; };
    auto values_PPPO = [](uint i, uint j) { return arr{arr{0.8_R, 0.8_R}, arr{0.8_R, 0.1_R}}[i][j]; };

    auto values_PPOO = [](uint i, uint j) { return arr{arr{0.8_R, 0.8_R}, arr{0.1_R, 0.1_R}}[i][j]; };
    auto values_OOPP = [](uint i, uint j) { return arr{arr{0.1_R, 0.1_R}, arr{0.8_R, 0.8_R}}[i][j]; };

    auto values_POPO = [](uint i, uint j) { return arr{arr{0.8_R, 0.1_R}, arr{0.8_R, 0.1_R}}[i][j]; };
    auto values_OPOP = [](uint i, uint j) { return arr{arr{0.1_R, 0.8_R}, arr{0.1_R, 0.8_R}}[i][j]; };

    Real isoValue = 0.4_R;

    testExtract_AAAA(positions, values_OOOO, isoValue);
    testExtract_AAAA(positions, values_PPPP, isoValue);

    testExtract_BAAA(positions, values_POOO, isoValue);
    testExtract_BAAA(positions, values_OPPP, isoValue);

    testExtract_ABAA(positions, values_OPOO, isoValue);
    testExtract_ABAA(positions, values_POPP, isoValue);

    testExtract_AABA(positions, values_OOPO, isoValue);
    testExtract_AABA(positions, values_PPOP, isoValue);

    testExtract_AAAB(positions, values_OOOP, isoValue);
    testExtract_AAAB(positions, values_PPPO, isoValue);

    testExtract_BBAA(positions, values_PPOO, isoValue);
    testExtract_BBAA(positions, values_OOPP, isoValue);

    testExtract_BABA(positions, values_POPO, isoValue);
    testExtract_BABA(positions, values_OPOP, isoValue);
  }
}

} // namespace ARIA
