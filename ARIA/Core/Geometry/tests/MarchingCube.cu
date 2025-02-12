#include "ARIA/MarchingCube.h"

#include <gtest/gtest.h>

#include <cuda/std/span>

namespace ARIA {

namespace {

template <typename T, size_t n>
using arr = std::array<T, n>;

ARIA_HOST_DEVICE static inline Real ComputeT(Real v0, Real v1, Real isoValue) {
  return (isoValue - v0) / (v1 - v0);
}

static inline void SortPrimitiveVertices(const cuda::std::span<Vec2r, 2> &primitiveVertices) {
  std::ranges::sort(primitiveVertices,
                    [](const Vec2r &a, const Vec2r &b) { return a.y() < b.y() || (a.y() == b.y() && a.x() < b.x()); });
};

static inline void SortPrimitiveVertices(const cuda::std::span<Vec3r> &primitiveVertices) {
  std::ranges::sort(primitiveVertices, [](const Vec3r &a, const Vec3r &b) {
    return a.z() < b.z() || (a.z() == b.z() && (a.y() < b.y() || (a.y() == b.y() && a.x() < b.x())));
  });
};

static inline void ExpectEq(const Vec1r &a, const Vec1r &b) {
  EXPECT_FLOAT_EQ(a.x(), b.x());
}

static inline void ExpectEq(const Vec2r &a, const Vec2r &b) {
  EXPECT_FLOAT_EQ(a.x(), b.x());
  EXPECT_FLOAT_EQ(a.y(), b.y());
}

static inline void ExpectEq(const Vec3r &a, const Vec3r &b) {
  EXPECT_FLOAT_EQ(a.x(), b.x());
  EXPECT_FLOAT_EQ(a.y(), b.y());
  EXPECT_FLOAT_EQ(a.z(), b.z());
}

} // namespace

TEST(MarchingCube, D1) {
  using MC = MarchingCube<1>;

  auto testExtract_AA = [&](const auto &positions, const auto &values, Real isoValue) {
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec1r, 1> &) { EXPECT_FALSE(true); });
  };

  auto testExtract_BA = [&](const auto &positions, const auto &values, Real isoValue) {
    uint times = 0;
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec1r, 1> &primitiveVertices) {
      EXPECT_EQ(times, 0);
      Vec1r p = Lerp(positions(0), positions(1), ComputeT(values(0), values(1), isoValue));
      ExpectEq(primitiveVertices[0], p);
      ++times;
    });
    EXPECT_EQ(times, 1);
  };

  auto positions = [](uint i) { return arr{Vec1r{-2.5_R}, Vec1r{3.75_R}}[i]; };

  auto valuesOO = [](uint i) { return arr{0.1_R, 0.1_R}[i]; };
  auto valuesPP = [](uint i) { return arr{0.8_R, 0.8_R}[i]; };

  auto valuesPO = [](uint i) { return arr{0.8_R, 0.1_R}[i]; };
  auto valuesOP = [](uint i) { return arr{0.1_R, 0.8_R}[i]; };

  Real isoValue = 0.4_R;

  testExtract_AA(positions, valuesOO, isoValue);
  testExtract_AA(positions, valuesPP, isoValue);

  testExtract_BA(positions, valuesPO, isoValue);
  testExtract_BA(positions, valuesOP, isoValue);
}

TEST(MarchingCube, D2) {
  using MC = MarchingCube<2>;

  auto testExtract_AAAA = [&](const auto &positions, const auto &values, Real isoValue) {
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &) { EXPECT_FALSE(true); });
  };

  auto testExtract_BAAA = [&](const auto &positions, const auto &values, Real isoValue) {
    uint times = 0;
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
      EXPECT_EQ(times, 0);
      SortPrimitiveVertices(primitiveVertices);
      Vec2r p_00_10 = Lerp(positions(0, 0), positions(1, 0), ComputeT(values(0, 0), values(1, 0), isoValue));
      Vec2r p_00_01 = Lerp(positions(0, 0), positions(0, 1), ComputeT(values(0, 0), values(0, 1), isoValue));
      ExpectEq(primitiveVertices[0], p_00_10);
      ExpectEq(primitiveVertices[1], p_00_01);
      ++times;
    });
    EXPECT_EQ(times, 1);
  };

  auto testExtract_ABAA = [&](const auto &positions, const auto &values, Real isoValue) {
    uint times = 0;
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
      EXPECT_EQ(times, 0);
      SortPrimitiveVertices(primitiveVertices);
      Vec2r p_00_01 = Lerp(positions(0, 0), positions(0, 1), ComputeT(values(0, 0), values(0, 1), isoValue));
      Vec2r p_01_11 = Lerp(positions(0, 1), positions(1, 1), ComputeT(values(0, 1), values(1, 1), isoValue));
      ExpectEq(primitiveVertices[0], p_00_01);
      ExpectEq(primitiveVertices[1], p_01_11);
      ++times;
    });
    EXPECT_EQ(times, 1);
  };

  auto testExtract_AABA = [&](const auto &positions, const auto &values, Real isoValue) {
    uint times = 0;
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
      EXPECT_EQ(times, 0);
      SortPrimitiveVertices(primitiveVertices);
      Vec2r p_00_10 = Lerp(positions(0, 0), positions(1, 0), ComputeT(values(0, 0), values(1, 0), isoValue));
      Vec2r p_10_11 = Lerp(positions(1, 0), positions(1, 1), ComputeT(values(1, 0), values(1, 1), isoValue));
      ExpectEq(primitiveVertices[0], p_00_10);
      ExpectEq(primitiveVertices[1], p_10_11);
      ++times;
    });
    EXPECT_EQ(times, 1);
  };

  auto testExtract_AAAB = [&](const auto &positions, const auto &values, Real isoValue) {
    uint times = 0;
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
      EXPECT_EQ(times, 0);
      SortPrimitiveVertices(primitiveVertices);
      Vec2r p_10_11 = Lerp(positions(1, 0), positions(1, 1), ComputeT(values(1, 0), values(1, 1), isoValue));
      Vec2r p_01_11 = Lerp(positions(0, 1), positions(1, 1), ComputeT(values(0, 1), values(1, 1), isoValue));
      ExpectEq(primitiveVertices[0], p_10_11);
      ExpectEq(primitiveVertices[1], p_01_11);
      ++times;
    });
    EXPECT_EQ(times, 1);
  };

  auto testExtract_BBAA = [&](const auto &positions, const auto &values, Real isoValue) {
    uint times = 0;
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
      EXPECT_EQ(times, 0);
      SortPrimitiveVertices(primitiveVertices);
      Vec2r p_00_10 = Lerp(positions(0, 0), positions(1, 0), ComputeT(values(0, 0), values(1, 0), isoValue));
      Vec2r p_01_11 = Lerp(positions(0, 1), positions(1, 1), ComputeT(values(0, 1), values(1, 1), isoValue));
      ExpectEq(primitiveVertices[0], p_00_10);
      ExpectEq(primitiveVertices[1], p_01_11);
      ++times;
    });
    EXPECT_EQ(times, 1);
  };

  auto testExtract_BABA = [&](const auto &positions, const auto &values, Real isoValue) {
    uint times = 0;
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
      EXPECT_EQ(times, 0);
      SortPrimitiveVertices(primitiveVertices);
      Vec2r p_00_01 = Lerp(positions(0, 0), positions(0, 1), ComputeT(values(0, 0), values(0, 1), isoValue));
      Vec2r p_10_11 = Lerp(positions(1, 0), positions(1, 1), ComputeT(values(1, 0), values(1, 1), isoValue));
      ExpectEq(primitiveVertices[0], p_00_01);
      ExpectEq(primitiveVertices[1], p_10_11);
      ++times;
    });
    EXPECT_EQ(times, 1);
  };

  auto testExtract_POOP = [&](const auto &positions, const auto &values, Real isoValue) {
    uint times = 0;
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
      EXPECT_TRUE(times == 0 || times == 1);
      SortPrimitiveVertices(primitiveVertices);
      if (times == 0) {
        Vec2r p_00_10 = Lerp(positions(0, 0), positions(1, 0), ComputeT(values(0, 0), values(1, 0), isoValue));
        Vec2r p_00_01 = Lerp(positions(0, 0), positions(0, 1), ComputeT(values(0, 0), values(0, 1), isoValue));
        ExpectEq(primitiveVertices[0], p_00_10);
        ExpectEq(primitiveVertices[1], p_00_01);
      } else if (times == 1) {
        Vec2r p_10_11 = Lerp(positions(1, 0), positions(1, 1), ComputeT(values(1, 0), values(1, 1), isoValue));
        Vec2r p_01_11 = Lerp(positions(0, 1), positions(1, 1), ComputeT(values(0, 1), values(1, 1), isoValue));
        ExpectEq(primitiveVertices[0], p_10_11);
        ExpectEq(primitiveVertices[1], p_01_11);
      }
      ++times;
    });
    EXPECT_EQ(times, 2);
  };

  auto testExtract_OPPO = [&](const auto &positions, const auto &values, Real isoValue) {
    uint times = 0;
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec2r, 2> &primitiveVertices) {
      EXPECT_TRUE(times == 0 || times == 1);
      SortPrimitiveVertices(primitiveVertices);
      if (times == 0) {
        Vec2r p_00_10 = Lerp(positions(0, 0), positions(1, 0), ComputeT(values(0, 0), values(1, 0), isoValue));
        Vec2r p_10_11 = Lerp(positions(1, 0), positions(1, 1), ComputeT(values(1, 0), values(1, 1), isoValue));
        ExpectEq(primitiveVertices[0], p_00_10);
        ExpectEq(primitiveVertices[1], p_10_11);
      } else if (times == 1) {
        Vec2r p_00_01 = Lerp(positions(0, 0), positions(0, 1), ComputeT(values(0, 0), values(0, 1), isoValue));
        Vec2r p_01_11 = Lerp(positions(0, 1), positions(1, 1), ComputeT(values(0, 1), values(1, 1), isoValue));
        ExpectEq(primitiveVertices[0], p_00_01);
        ExpectEq(primitiveVertices[1], p_01_11);
      }
      ++times;
    });
    EXPECT_EQ(times, 2);
  };

  auto positions = [](uint i, uint j) {
    return arr{arr{Vec2r{-2.5_R, -2.5_R}, Vec2r{-2.5_R, 3.75_R}}, //
               arr{Vec2r{3.75_R, -2.5_R}, Vec2r{3.75_R, 3.75_R}}}[i][j];
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

  auto values_POOP = [](uint i, uint j) { return arr{arr{0.8_R, 0.1_R}, arr{0.1_R, 0.8_R}}[i][j]; };
  auto values_OPPO = [](uint i, uint j) { return arr{arr{0.1_R, 0.8_R}, arr{0.8_R, 0.1_R}}[i][j]; };

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

  testExtract_POOP(positions, values_POOP, isoValue);
  testExtract_OPPO(positions, values_OPPO, isoValue);
}

TEST(MarchingCube, D3) {
  using MC = MarchingCube<3>;

  auto extractAndGatherVertices = [](const auto &positions, const auto &values, Real isoValue) {
    uint times = 0;
    std::vector<Vec3r> vertices;
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec3r, 3> &primitiveVertices) {
      EXPECT_TRUE(times == 0 || times == 1);
      vertices.emplace_back(primitiveVertices[0]);
      vertices.emplace_back(primitiveVertices[1]);
      vertices.emplace_back(primitiveVertices[2]);
      ++times;
    });
    EXPECT_EQ(vertices.size(), 6);
    SortPrimitiveVertices(vertices);
    EXPECT_EQ(times, 2);
    return vertices;
  };

  auto testExtract_AAAA = [&](const auto &positions, const auto &values, Real isoValue) {
    MC::Extract(positions, values, isoValue, [&](const cuda::std::span<Vec3r, 3> &) { EXPECT_FALSE(true); });
  };

  auto testExtract_BAAA = [&](const auto &positions, const auto &values, Real isoValue) {
    std::vector vertices = extractAndGatherVertices(positions, values, isoValue);

    Vec3r p_000_100 =
        Lerp(positions(0, 0, 0), positions(1, 0, 0), ComputeT(values(0, 0, 0), values(1, 0, 0), isoValue));
    Vec3r p_000_010 =
        Lerp(positions(0, 0, 0), positions(0, 1, 0), ComputeT(values(0, 0, 0), values(0, 1, 0), isoValue));
    Vec3r p_001_101 =
        Lerp(positions(0, 0, 1), positions(1, 0, 1), ComputeT(values(0, 0, 1), values(1, 0, 1), isoValue));
    Vec3r p_001_011 =
        Lerp(positions(0, 0, 1), positions(0, 1, 1), ComputeT(values(0, 0, 1), values(0, 1, 1), isoValue));

    ExpectEq(vertices[0], p_000_100);
    ExpectEq(vertices[1], p_000_010);
    ExpectEq(vertices[2], p_000_010);
    ExpectEq(vertices[3], p_001_101);
    ExpectEq(vertices[4], p_001_101);
    ExpectEq(vertices[5], p_001_011);
  };

  auto testExtract_ABAA = [&](const auto &positions, const auto &values, Real isoValue) {
    std::vector vertices = extractAndGatherVertices(positions, values, isoValue);

    Vec3r p_000_010 =
        Lerp(positions(0, 0, 0), positions(0, 1, 0), ComputeT(values(0, 0, 0), values(0, 1, 0), isoValue));
    Vec3r p_010_110 =
        Lerp(positions(0, 1, 0), positions(1, 1, 0), ComputeT(values(0, 1, 0), values(1, 1, 0), isoValue));
    Vec3r p_001_011 =
        Lerp(positions(0, 0, 1), positions(0, 1, 1), ComputeT(values(0, 0, 1), values(0, 1, 1), isoValue));
    Vec3r p_011_111 =
        Lerp(positions(0, 1, 1), positions(1, 1, 1), ComputeT(values(0, 1, 1), values(1, 1, 1), isoValue));

    ExpectEq(vertices[0], p_000_010);
    ExpectEq(vertices[1], p_010_110);
    ExpectEq(vertices[2], p_010_110);
    ExpectEq(vertices[3], p_001_011);
    ExpectEq(vertices[4], p_001_011);
    ExpectEq(vertices[5], p_011_111);
  };

  auto positions = [](uint i, uint j, uint k) {
    return Vec3r{
        i == 0 ? -0.25_R : 3.75_R,
        j == 0 ? -0.25_R : 3.75_R,
        k == 0 ? -0.25_R : 3.75_R,
    };
  };

  auto values_OOOO = [](uint i, uint j, uint k) { return arr{arr{0.1_R, 0.1_R}, arr{0.1_R, 0.1_R}}[i][j]; };
  auto values_PPPP = [](uint i, uint j, uint k) { return arr{arr{0.8_R, 0.8_R}, arr{0.8_R, 0.8_R}}[i][j]; };

  auto values_POOO = [](uint i, uint j, uint k) { return arr{arr{0.8_R, 0.1_R}, arr{0.1_R, 0.1_R}}[i][j]; };
  auto values_OPPP = [](uint i, uint j, uint k) { return arr{arr{0.1_R, 0.8_R}, arr{0.8_R, 0.8_R}}[i][j]; };

  auto values_OPOO = [](uint i, uint j, uint k) { return arr{arr{0.1_R, 0.8_R}, arr{0.1_R, 0.1_R}}[i][j]; };
  auto values_POPP = [](uint i, uint j, uint k) { return arr{arr{0.8_R, 0.1_R}, arr{0.8_R, 0.8_R}}[i][j]; };

  auto values_OOPO = [](uint i, uint j, uint k) { return arr{arr{0.1_R, 0.1_R}, arr{0.8_R, 0.1_R}}[i][j]; };
  auto values_PPOP = [](uint i, uint j, uint k) { return arr{arr{0.8_R, 0.8_R}, arr{0.1_R, 0.8_R}}[i][j]; };

  auto values_OOOP = [](uint i, uint j, uint k) { return arr{arr{0.1_R, 0.1_R}, arr{0.1_R, 0.8_R}}[i][j]; };
  auto values_PPPO = [](uint i, uint j, uint k) { return arr{arr{0.8_R, 0.8_R}, arr{0.8_R, 0.1_R}}[i][j]; };

  auto values_PPOO = [](uint i, uint j, uint k) { return arr{arr{0.8_R, 0.8_R}, arr{0.1_R, 0.1_R}}[i][j]; };
  auto values_OOPP = [](uint i, uint j, uint k) { return arr{arr{0.1_R, 0.1_R}, arr{0.8_R, 0.8_R}}[i][j]; };

  auto values_POPO = [](uint i, uint j, uint k) { return arr{arr{0.8_R, 0.1_R}, arr{0.8_R, 0.1_R}}[i][j]; };
  auto values_OPOP = [](uint i, uint j, uint k) { return arr{arr{0.1_R, 0.8_R}, arr{0.1_R, 0.8_R}}[i][j]; };

  auto values_POOP = [](uint i, uint j, uint k) { return arr{arr{0.8_R, 0.1_R}, arr{0.1_R, 0.8_R}}[i][j]; };
  auto values_OPPO = [](uint i, uint j, uint k) { return arr{arr{0.1_R, 0.8_R}, arr{0.8_R, 0.1_R}}[i][j]; };

  Real isoValue = 0.4_R;

  testExtract_AAAA(positions, values_OOOO, isoValue);
  testExtract_AAAA(positions, values_PPPP, isoValue);

  testExtract_BAAA(positions, values_POOO, isoValue);
  testExtract_BAAA(positions, values_OPPP, isoValue);

  testExtract_ABAA(positions, values_OPOO, isoValue);
  testExtract_ABAA(positions, values_POPP, isoValue);
}

} // namespace ARIA
