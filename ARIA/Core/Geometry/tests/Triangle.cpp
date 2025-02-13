#include "ARIA/Triangle.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(Triangle, Base) {
  // 2D.
  {
    Triangle2f triDefaultConstructed;
    Triangle2f tri{{0.0F, 0.0F}, {1.0F, 0.0F}, {0.0F, 1.0F}};
    Vec2f v0{0.0F, 0.0F}, v1{1.0F, 0.0F}, v2{0.0F, 1.0F};
    EXPECT_EQ(tri[0], v0);
    EXPECT_EQ(tri[1], v1);
    EXPECT_EQ(tri[2], v2);
  }

  // 3D.
  {
    Triangle3f triDefaultConstructed;
    Triangle3f tri{{0.0F, 0.0F, 0.0F}, {1.0F, 0.0F, 0.0F}, {0.0F, 1.0F, 0.0F}};
    Vec3f v0{0.0F, 0.0F, 0.0F}, v1{1.0F, 0.0F, 0.0F}, v2{0.0F, 1.0F, 0.0F};
    EXPECT_EQ(tri[0], v0);
    EXPECT_EQ(tri[1], v1);
    EXPECT_EQ(tri[2], v2);

    Vec3f normal = tri.normal();
    EXPECT_FLOAT_EQ(normal.x(), 0);
    EXPECT_FLOAT_EQ(normal.y(), 0);
    EXPECT_FLOAT_EQ(normal.z(), 1);

    Vec3r stableNormal = tri.stableNormal();
    EXPECT_FLOAT_EQ(stableNormal.x(), 0);
    EXPECT_FLOAT_EQ(stableNormal.y(), 0);
    EXPECT_FLOAT_EQ(stableNormal.z(), 1);

    Vec3r stableNormalZeros = Triangle3f{{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}}.stableNormal();
    EXPECT_FLOAT_EQ(stableNormalZeros.x(), 0);
    EXPECT_FLOAT_EQ(stableNormalZeros.y(), 0);
    EXPECT_FLOAT_EQ(stableNormalZeros.z(), 0);
  }
}

} // namespace ARIA
