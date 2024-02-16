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

  auto expectAABB3 = [](const AABB3f &aabb, const Vec3f &inf, const Vec3f &sup) {
    EXPECT_FLOAT_EQ(aabb.inf().x(), inf.x());
    EXPECT_FLOAT_EQ(aabb.inf().y(), inf.y());
    EXPECT_FLOAT_EQ(aabb.inf().z(), inf.z());
    EXPECT_FLOAT_EQ(aabb.sup().x(), sup.x());
    EXPECT_FLOAT_EQ(aabb.sup().y(), sup.y());
    EXPECT_FLOAT_EQ(aabb.sup().z(), sup.z());
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
    expectAABB3(aabb0, {infinity<float>, infinity<float>, infinity<float>},
                {-infinity<float>, -infinity<float>, -infinity<float>});

    // Copy constructor.
    AABB3f aabb1 = aabb0;
    expectAABB3(aabb1, {infinity<float>, infinity<float>, infinity<float>},
                {-infinity<float>, -infinity<float>, -infinity<float>});

    // Construct from 1 point.
    AABB3f aabb2{Vec3f{0.1F, 0.1F, 0.1F}};
    expectAABB3(aabb2, {0.1F, 0.1F, 0.1F}, {0.1F, 0.1F, 0.1F});

    // Construct from 2 AABBs.
    AABB3f aabb3{Vec3f{0.01F, 0.01F, 0.01F}};
    AABB3f aabb4{aabb2, aabb3};
    expectAABB3(aabb4, {0.01F, 0.01F, 0.01F}, {0.1F, 0.1F, 0.1F});

    // Construct from 1 AABB + 1 point.
    AABB3f aabb5{aabb4, Vec3f{0.001F, 0.001F, 0.001F}};
    expectAABB3(aabb5, {0.001F, 0.001F, 0.001F}, {0.1F, 0.1F, 0.1F});

    // Construct from 3 AABBs.
    AABB3f aabb6{aabb5, aabb4, aabb3};
    expectAABB3(aabb6, {0.001F, 0.001F, 0.001F}, {0.1F, 0.1F, 0.1F});
  }
}

} // namespace ARIA
