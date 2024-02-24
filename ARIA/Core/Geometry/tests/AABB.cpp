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

  // 2D constructors, unionize, inf, sup, and empty.
  {
    // Default constructor.
    AABB2f aabb0;
    expectAABB2(aabb0, {infinity<float>, infinity<float>}, {-infinity<float>, -infinity<float>});
    EXPECT_TRUE(aabb0.empty());

    // Copy constructor.
    AABB2f aabb1 = aabb0;
    expectAABB2(aabb1, {infinity<float>, infinity<float>}, {-infinity<float>, -infinity<float>});
    EXPECT_TRUE(aabb1.empty());

    // Construct from 1 point.
    AABB2f aabb2{Vec2f{0.1F, 0.1F}};
    expectAABB2(aabb2, {0.1F, 0.1F}, {0.1F, 0.1F});
    EXPECT_FALSE(aabb2.empty());

    // Construct from 2 AABBs.
    AABB2f aabb3{Vec2f{0.01F, 0.01F}};
    AABB2f aabb4{aabb2, aabb3};
    expectAABB2(aabb4, {0.01F, 0.01F}, {0.1F, 0.1F});
    EXPECT_FALSE(aabb4.empty());

    // Construct from 1 AABB + 1 point.
    AABB2f aabb5{aabb4, Vec2f{0.001F, 0.001F}};
    expectAABB2(aabb5, {0.001F, 0.001F}, {0.1F, 0.1F});
    EXPECT_FALSE(aabb5.empty());

    // Construct from 3 items.
    Vec2f point{0.001F, 0.001F};
    {
      AABB2f aabb{aabb5, aabb4, aabb3};
      expectAABB2(aabb, {0.001F, 0.001F}, {0.1F, 0.1F});
    }
    {
      AABB2f aabb{aabb5, aabb4, point};
      expectAABB2(aabb, {0.001F, 0.001F}, {0.1F, 0.1F});
    }
    {
      AABB2f aabb{aabb5, point, aabb3};
      expectAABB2(aabb, {0.001F, 0.001F}, {0.1F, 0.1F});
    }
    {
      AABB2f aabb{aabb5, point, point};
      expectAABB2(aabb, {0.001F, 0.001F}, {0.1F, 0.1F});
    }
    {
      AABB2f aabb{point, aabb4, aabb3};
      expectAABB2(aabb, {0.001F, 0.001F}, {0.1F, 0.1F});
    }
    {
      AABB2f aabb{point, aabb4, point};
      expectAABB2(aabb, {0.001F, 0.001F}, {0.1F, 0.1F});
    }
    {
      AABB2f aabb{point, point, aabb3};
      expectAABB2(aabb, {0.001F, 0.001F}, {0.01F, 0.01F});
    }
    {
      AABB2f aabb{point, point, point};
      expectAABB2(aabb, {0.001F, 0.001F}, {0.001F, 0.001F});
    }

    AABB2f aabb6 = aabb2;
    aabb6.Unionize(Vec2f{0.05F, 0.05F}, Vec2f{0.01F, 0.01F});
    expectAABB2(aabb6, {0.01F, 0.01F}, {0.1F, 0.1F});

    // Complex Empty.
    {
      AABB2f aabb;
      aabb.inf() = {0.1F, 0.2F};
      aabb.sup() = {0.3F, 0.4F};
      EXPECT_FALSE(aabb.empty());
    }

    {
      AABB2f aabb;
      aabb.inf() = {0.1F, 0.4F};
      aabb.sup() = {0.3F, 0.2F};
      EXPECT_TRUE(aabb.empty());
    }

    {
      AABB2f aabb;
      aabb.inf() = {0.3F, 0.2F};
      aabb.sup() = {0.1F, 0.4F};
      EXPECT_TRUE(aabb.empty());
    }

    {
      AABB2f aabb;
      aabb.inf() = {0.3F, 0.4F};
      aabb.sup() = {0.1F, 0.2F};
      EXPECT_TRUE(aabb.empty());
    }
  }

  // 3D constructors, unionize, inf, sup, and empty.
  {
    // Default constructor.
    AABB3f aabb0;
    expectAABB3(aabb0, {infinity<float>, infinity<float>, infinity<float>},
                {-infinity<float>, -infinity<float>, -infinity<float>});
    EXPECT_TRUE(aabb0.empty());

    // Copy constructor.
    AABB3f aabb1 = aabb0;
    expectAABB3(aabb1, {infinity<float>, infinity<float>, infinity<float>},
                {-infinity<float>, -infinity<float>, -infinity<float>});
    EXPECT_TRUE(aabb1.empty());

    // Construct from 1 point.
    AABB3f aabb2{Vec3f{0.1F, 0.1F, 0.1F}};
    expectAABB3(aabb2, {0.1F, 0.1F, 0.1F}, {0.1F, 0.1F, 0.1F});
    EXPECT_FALSE(aabb2.empty());

    // Construct from 2 AABBs.
    AABB3f aabb3{Vec3f{0.01F, 0.01F, 0.01F}};
    AABB3f aabb4{aabb2, aabb3};
    expectAABB3(aabb4, {0.01F, 0.01F, 0.01F}, {0.1F, 0.1F, 0.1F});
    EXPECT_FALSE(aabb4.empty());

    // Construct from 1 AABB + 1 point.
    AABB3f aabb5{aabb4, Vec3f{0.001F, 0.001F, 0.001F}};
    expectAABB3(aabb5, {0.001F, 0.001F, 0.001F}, {0.1F, 0.1F, 0.1F});
    EXPECT_FALSE(aabb5.empty());

    // Construct from 3 items.
    Vec3f point{0.001F, 0.001F, 0.001F};
    {
      AABB3f aabb{aabb5, aabb4, aabb3};
      expectAABB3(aabb, {0.001F, 0.001F, 0.001F}, {0.1F, 0.1F, 0.1F});
    }
    {
      AABB3f aabb{aabb5, aabb4, point};
      expectAABB3(aabb, {0.001F, 0.001F, 0.001F}, {0.1F, 0.1F, 0.1F});
    }
    {
      AABB3f aabb{aabb5, point, aabb3};
      expectAABB3(aabb, {0.001F, 0.001F, 0.001F}, {0.1F, 0.1F, 0.1F});
    }
    {
      AABB3f aabb{aabb5, point, point};
      expectAABB3(aabb, {0.001F, 0.001F, 0.001F}, {0.1F, 0.1F, 0.1F});
    }
    {
      AABB3f aabb{point, aabb4, aabb3};
      expectAABB3(aabb, {0.001F, 0.001F, 0.001F}, {0.1F, 0.1F, 0.1F});
    }
    {
      AABB3f aabb{point, aabb4, point};
      expectAABB3(aabb, {0.001F, 0.001F, 0.001F}, {0.1F, 0.1F, 0.1F});
    }
    {
      AABB3f aabb{point, point, aabb3};
      expectAABB3(aabb, {0.001F, 0.001F, 0.001F}, {0.01F, 0.01F, 0.01F});
    }
    {
      AABB3f aabb{point, point, point};
      expectAABB3(aabb, {0.001F, 0.001F, 0.001F}, {0.001F, 0.001F, 0.001F});
    }

    AABB3f aabb6 = aabb2;
    aabb6.Unionize(Vec3f{0.05F, 0.05F, 0.05F}, Vec3f{0.01F, 0.01F, 0.01F});
    expectAABB3(aabb6, {0.01F, 0.01F, 0.01F}, {0.1F, 0.1F, 0.1F});

    // Complex Empty.
    {
      AABB3f aabb;
      aabb.inf() = {0.1F, 0.2F, 0.3F};
      aabb.sup() = {0.4F, 0.5F, 0.6F};
      EXPECT_FALSE(aabb.empty());
    }

    {
      AABB3f aabb;
      aabb.inf() = {0.1F, 0.2F, 0.6F};
      aabb.sup() = {0.4F, 0.5F, 0.3F};
      EXPECT_TRUE(aabb.empty());
    }

    {
      AABB3f aabb;
      aabb.inf() = {0.1F, 0.5F, 0.3F};
      aabb.sup() = {0.4F, 0.2F, 0.6F};
      EXPECT_TRUE(aabb.empty());
    }

    {
      AABB3f aabb;
      aabb.inf() = {0.1F, 0.5F, 0.6F};
      aabb.sup() = {0.4F, 0.2F, 0.3F};
      EXPECT_TRUE(aabb.empty());
    }

    {
      AABB3f aabb;
      aabb.inf() = {0.4F, 0.2F, 0.3F};
      aabb.sup() = {0.1F, 0.5F, 0.6F};
      EXPECT_TRUE(aabb.empty());
    }

    {
      AABB3f aabb;
      aabb.inf() = {0.4F, 0.2F, 0.6F};
      aabb.sup() = {0.1F, 0.5F, 0.3F};
      EXPECT_TRUE(aabb.empty());
    }

    {
      AABB3f aabb;
      aabb.inf() = {0.4F, 0.5F, 0.3F};
      aabb.sup() = {0.1F, 0.2F, 0.6F};
      EXPECT_TRUE(aabb.empty());
    }

    {
      AABB3f aabb;
      aabb.inf() = {0.4F, 0.5F, 0.6F};
      aabb.sup() = {0.1F, 0.2F, 0.3F};
      EXPECT_TRUE(aabb.empty());
    }
  }

  // Is AABB.
  {
    static_assert(aabb::detail::is_aabb_v<AABB2i>);
    static_assert(aabb::detail::is_aabb_d_v<AABB2i, 2>);
    static_assert(!aabb::detail::is_aabb_d_v<AABB2i, 1>);
    static_assert(!aabb::detail::is_aabb_d_v<AABB2i, 3>);
  }
}

TEST(AABB, Methods) {
  auto expectV2 = [](const Vec2r &lhs, const Vec2r &rhs) {
    EXPECT_FLOAT_EQ(float(lhs.x()), float(rhs.x()));
    EXPECT_FLOAT_EQ(float(lhs.y()), float(rhs.y()));
  };

  auto expectV3 = [](const Vec3r &lhs, const Vec3r &rhs) {
    EXPECT_FLOAT_EQ(float(lhs.x()), float(rhs.x()));
    EXPECT_FLOAT_EQ(float(lhs.y()), float(rhs.y()));
    EXPECT_FLOAT_EQ(float(lhs.z()), float(rhs.z()));
  };

  // 2D other methods.
  {
    AABB2f aabb;
    aabb.inf() = {0.1F, 0.2F};
    aabb.sup() = {0.3F, 0.5F};

    expectV2(aabb.center(), {0.2F, 0.35F});
    expectV2(aabb.offset({0.25F, 0.4F}), {0.75F, 0.666666666666667F});
    expectV2(aabb.diagonal(), {0.2F, 0.3F});

    expectV2(aabb[0], {0.1F, 0.2F});
    expectV2(aabb[1], {0.3F, 0.5F});

    expectV2(aabb[C<0>{}], {0.1F, 0.2F});
    expectV2(aabb[C<1>{}], {0.3F, 0.5F});
  }

  // 3D other methods.
  {
    AABB3f aabb;
    aabb.inf() = {0.1F, 0.2F, 0.3F};
    aabb.sup() = {0.4F, 0.6F, 0.9F};

    expectV3(aabb.center(), {0.25F, 0.4F, 0.6F});
    expectV3(aabb.offset({0.25F, 0.5F, 0.8F}), {0.5F, 0.75F, 0.833333333333333F});
    expectV3(aabb.diagonal(), {0.3F, 0.4F, 0.6F});

    expectV3(aabb[0], {0.1F, 0.2F, 0.3F});
    expectV3(aabb[1], {0.4F, 0.6F, 0.9F});

    expectV3(aabb[C<0>{}], {0.1F, 0.2F, 0.3F});
    expectV3(aabb[C<1>{}], {0.4F, 0.6F, 0.9F});
  }
}

} // namespace ARIA
