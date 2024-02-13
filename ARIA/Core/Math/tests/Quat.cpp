#include "ARIA/Quat.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

class Test {
public:
  ARIA_PROP_PREFAB_QUAT(public, public, , Quatr, rotation);
};

} // namespace

TEST(Quat, Base) {
  auto expectV = [](const Vec3r &lhs, const Vec3r &rhs) {
    EXPECT_FLOAT_EQ(float(lhs.x()), float(rhs.x()));
    EXPECT_FLOAT_EQ(float(lhs.y()), float(rhs.y()));
    EXPECT_FLOAT_EQ(float(lhs.z()), float(rhs.z()));
  };

  auto expectQ = [](const Quatr &lhs, const Quatr &rhs) {
    EXPECT_FLOAT_EQ(float(lhs.w()), float(rhs.w()));
    EXPECT_FLOAT_EQ(float(lhs.x()), float(rhs.x()));
    EXPECT_FLOAT_EQ(float(lhs.y()), float(rhs.y()));
    EXPECT_FLOAT_EQ(float(lhs.z()), float(rhs.z()));
  };

  // Length.
  {
    static_assert(sizeof(Quatf) == 4 * sizeof(float));
    static_assert(sizeof(Quatd) == 4 * sizeof(double));
    static_assert(sizeof(Quatr) == 4 * sizeof(Real));
  }

  // Is `Quat`.
  {
    static_assert(quat::detail::is_quat_v<Quatf>);
    static_assert(quat::detail::is_quat_v<Quatd>);
    static_assert(quat::detail::is_quat_v<Quatr>);
    static_assert(!quat::detail::is_quat_v<Vec4f>);
    static_assert(!quat::detail::is_quat_v<Vec4d>);
    static_assert(!quat::detail::is_quat_v<Vec4r>);
  }

  // WXYZ.
  {
    Quatf q{0.1, 1.2, 2.3, 3.4};
    EXPECT_FLOAT_EQ(float(q.w()), 0.1);
    EXPECT_FLOAT_EQ(float(q.x()), 1.2);
    EXPECT_FLOAT_EQ(float(q.y()), 2.3);
    EXPECT_FLOAT_EQ(float(q.z()), 3.4);
  }

  // From and to Euler angles.
  {
    Quatf q0 = Quatf::Identity(); // 1, 0, 0, 0.
    expectV(ToEulerAngles(q0), {0, 0, 0});

    Quatf q1 = FromEulerAngles(Vec3f(11.0F, 22.0F, 33.0F) * deg2Rad<float>);
    EXPECT_TRUE(std::abs(q1.w() - 0.942065F) < 1e-4F);
    EXPECT_TRUE(std::abs(q1.x() - 0.036267F) < 1e-4F);
    EXPECT_TRUE(std::abs(q1.y() - 0.208831F) < 1e-4F);
    EXPECT_TRUE(std::abs(q1.z() - 0.259979F) < 1e-4F);
  }
}

} // namespace ARIA
