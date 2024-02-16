#include "ARIA/AABB.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(AABB, Base) {
  auto expectAABB2 = [](const AABB2f &aabb, const Vec2f &inf, const Vec2f &sup) {
    EXPECT_FLOAT_EQ(aabb.inf().x(), inf.x());
    EXPECT_FLOAT_EQ(aabb.inf().y(), inf.y());
    EXPECT_FLOAT_EQ(aabb.sup().x(), sup.x());
    EXPECT_FLOAT_EQ(aabb.sup().y(), sup.y());
  };

  // 2D.
  {
    // Default constructor.
    AABB2f aabb0;
    expectAABB2(aabb0, {infinity<float>, infinity<float>}, {-infinity<float>, -infinity<float>});

    // Copy constructor.
    AABB2f aabb1 = aabb0;
    expectAABB2(aabb1, {infinity<float>, infinity<float>}, {-infinity<float>, -infinity<float>});

    // Construct from 1 point.
    AABB2f aabb2{Vec2f{0.1F, 0.1F}};
    expectAABB2(aabb2, {0.1F, 0.1F}, {0.1F, 0.1F});

    // Construct from 2 AABBs.
    AABB2f aabb3{Vec2f{0.01F, 0.01F}};
    AABB2f aabb4{aabb2, aabb3};
    expectAABB2(aabb4, {0.01F, 0.01F}, {0.1F, 0.1F});

    // Construct from 1 AABB + 1 point.
    AABB2f aabb5{aabb4, Vec2f{0.001F, 0.001F}};
    expectAABB2(aabb5, {0.001F, 0.001F}, {0.1F, 0.1F});

    // Construct from 3 AABBs.
    AABB2f aabb6{aabb5, aabb4, aabb3};
    expectAABB2(aabb6, {0.001F, 0.001F}, {0.1F, 0.1F});
  }

  // 3D.
  {
    // Default constructor.
    AABB3f aabb0;
    EXPECT_FLOAT_EQ(aabb0.inf().x(), infinity<float>);
    EXPECT_FLOAT_EQ(aabb0.inf().y(), infinity<float>);
    EXPECT_FLOAT_EQ(aabb0.inf().z(), infinity<float>);
    EXPECT_FLOAT_EQ(aabb0.sup().x(), -infinity<float>);
    EXPECT_FLOAT_EQ(aabb0.sup().y(), -infinity<float>);
    EXPECT_FLOAT_EQ(aabb0.sup().z(), -infinity<float>);

    // Copy constructor.
    AABB3f aabb1 = aabb0;
    EXPECT_FLOAT_EQ(aabb1.inf().x(), infinity<float>);
    EXPECT_FLOAT_EQ(aabb1.inf().y(), infinity<float>);
    EXPECT_FLOAT_EQ(aabb1.inf().z(), infinity<float>);
    EXPECT_FLOAT_EQ(aabb1.sup().x(), -infinity<float>);
    EXPECT_FLOAT_EQ(aabb1.sup().y(), -infinity<float>);
    EXPECT_FLOAT_EQ(aabb1.sup().z(), -infinity<float>);

    // Construct from 1 point.
    AABB3f aabb2{Vec3f{0.1F, 0.1F, 0.1F}};
    EXPECT_FLOAT_EQ(aabb2.inf().x(), 0.1F);
    EXPECT_FLOAT_EQ(aabb2.inf().y(), 0.1F);
    EXPECT_FLOAT_EQ(aabb2.inf().z(), 0.1F);
    EXPECT_FLOAT_EQ(aabb2.sup().x(), 0.1F);
    EXPECT_FLOAT_EQ(aabb2.sup().y(), 0.1F);
    EXPECT_FLOAT_EQ(aabb2.sup().z(), 0.1F);

    // Construct from 2 AABBs.
    AABB3f aabb3{Vec3f{0.01F, 0.01F, 0.01F}};
    AABB3f aabb4{aabb2, aabb3};
    EXPECT_FLOAT_EQ(aabb4.inf().x(), 0.01F);
    EXPECT_FLOAT_EQ(aabb4.inf().y(), 0.01F);
    EXPECT_FLOAT_EQ(aabb4.inf().z(), 0.01F);
    EXPECT_FLOAT_EQ(aabb4.sup().x(), 0.1F);
    EXPECT_FLOAT_EQ(aabb4.sup().y(), 0.1F);
    EXPECT_FLOAT_EQ(aabb4.sup().z(), 0.1F);

    // Construct from 1 AABB + 1 point.
    AABB3f aabb5{aabb4, Vec3f{0.001F, 0.001F, 0.001F}};
    EXPECT_FLOAT_EQ(aabb5.inf().x(), 0.001F);
    EXPECT_FLOAT_EQ(aabb5.inf().y(), 0.001F);
    EXPECT_FLOAT_EQ(aabb5.inf().z(), 0.001F);
    EXPECT_FLOAT_EQ(aabb5.sup().x(), 0.1F);
    EXPECT_FLOAT_EQ(aabb5.sup().y(), 0.1F);
    EXPECT_FLOAT_EQ(aabb5.sup().z(), 0.1F);

    // Construct from 3 AABBs.
    AABB3f aabb6{aabb5, aabb4, aabb3};
    EXPECT_FLOAT_EQ(aabb6.inf().x(), 0.001F);
    EXPECT_FLOAT_EQ(aabb6.inf().y(), 0.001F);
    EXPECT_FLOAT_EQ(aabb6.inf().z(), 0.001F);
    EXPECT_FLOAT_EQ(aabb6.sup().x(), 0.1F);
    EXPECT_FLOAT_EQ(aabb6.sup().y(), 0.1F);
    EXPECT_FLOAT_EQ(aabb6.sup().z(), 0.1F);
  }
}

} // namespace ARIA
