#include "ARIA/LineSegment.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(LineSegment, Base) {
  // 1D.
  {
    LineSegment1f lsDefaultConstructed;
    LineSegment1f ls{Vec1f{1.0F}, Vec1f{2.0F}};
    Vec1f v0{1.0F}, v1{2.0F};
    EXPECT_EQ(ls[0], v0);
    EXPECT_EQ(ls[1], v1);
  }

  // 2D.
  {
    LineSegment2f lsDefaultConstructed;
    LineSegment2f ls{Vec2f{1.0F, 1.2F}, Vec2f{2.0F, 2.3F}};
    Vec2f v0{1.0F, 1.2F}, v1{2.0F, 2.3F};
    EXPECT_EQ(ls[0], v0);
    EXPECT_EQ(ls[1], v1);
  }

  // 3D.
  {
    LineSegment3f lsDefaultConstructed;
    LineSegment3f ls{Vec3f{1.0F, 1.2F, 1.24F}, Vec3f{2.0F, 2.3F, 2.36F}};
    Vec3f v0{1.0F, 1.2F, 1.24F}, v1{2.0F, 2.3F, 2.36F};
    EXPECT_EQ(ls[0], v0);
    EXPECT_EQ(ls[1], v1);
  }
}

} // namespace ARIA
