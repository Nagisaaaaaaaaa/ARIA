#include "ARIA/MortonCode.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

template <typename I>
void TestMortonEncode2D() {
  using V = Vec2<I>;

  EXPECT_EQ(MortonCode::Encode(V{0, 0}), 0);
  EXPECT_EQ(MortonCode::Encode(V{1, 0}), 1);
  EXPECT_EQ(MortonCode::Encode(V{0, 1}), 2);
  EXPECT_EQ(MortonCode::Encode(V{1, 1}), 3);

  EXPECT_EQ(MortonCode::Encode(V{2, 0}), 4);
  EXPECT_EQ(MortonCode::Encode(V{3, 0}), 5);
  EXPECT_EQ(MortonCode::Encode(V{2, 1}), 6);
  EXPECT_EQ(MortonCode::Encode(V{3, 1}), 7);

  EXPECT_EQ(MortonCode::Encode(V{0, 2}), 8);
  EXPECT_EQ(MortonCode::Encode(V{1, 2}), 9);
  EXPECT_EQ(MortonCode::Encode(V{0, 3}), 10);
  EXPECT_EQ(MortonCode::Encode(V{1, 3}), 11);

  EXPECT_EQ(MortonCode::Encode(V{2, 2}), 12);
  EXPECT_EQ(MortonCode::Encode(V{3, 2}), 13);
  EXPECT_EQ(MortonCode::Encode(V{2, 3}), 14);
  EXPECT_EQ(MortonCode::Encode(V{3, 3}), 15);

  EXPECT_EQ(MortonCode::Encode(V{4, 0}), 16);
  EXPECT_EQ(MortonCode::Encode(V{5, 0}), 17);
  EXPECT_EQ(MortonCode::Encode(V{4, 1}), 18);
  EXPECT_EQ(MortonCode::Encode(V{5, 1}), 19);

  EXPECT_EQ(MortonCode::Encode(V{6, 0}), 20);
  EXPECT_EQ(MortonCode::Encode(V{7, 0}), 21);
  EXPECT_EQ(MortonCode::Encode(V{6, 1}), 22);
  EXPECT_EQ(MortonCode::Encode(V{7, 1}), 23);

  if constexpr (std::is_same_v<I, int>)
    EXPECT_EQ(MortonCode::Encode(V{0xFFFF, 0xFFFF}), -1);
  else
    EXPECT_EQ(MortonCode::Encode(V{0xFFFF, 0xFFFF}), 0xFFFFFFFF);

  if constexpr (sizeof(I) == 8) {
    if constexpr (std::is_same_v<I, int64>)
      EXPECT_EQ(MortonCode::Encode(V{0xFFFFFFFF, 0xFFFFFFFF}), -1);
    else
      EXPECT_EQ(MortonCode::Encode(V{0xFFFFFFFF, 0xFFFFFFFF}), 0xFFFFFFFFFFFFFFFFLLU);
  }
}

template <typename I>
void TestMortonEncode3D() {
  using V = Vec3<I>;

  EXPECT_EQ(MortonCode::Encode(V{0, 0, 0}), 0);
}

} // namespace

TEST(MortonCode, Base) {
  TestMortonEncode2D<int>();
  TestMortonEncode2D<uint>();
  TestMortonEncode2D<int64>();
  TestMortonEncode2D<uint64>();

  TestMortonEncode3D<int>();
  TestMortonEncode3D<uint>();
  TestMortonEncode3D<int64>();
  TestMortonEncode3D<uint64>();
}

} // namespace ARIA
