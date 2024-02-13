#include "ARIA/Scene/Components/Camera.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {} // namespace

TEST(Camera, Base) {
  auto expectV = [](const Vec3r &lhs, const Vec3r &rhs) {
    EXPECT_FLOAT_EQ(float(lhs.x()), float(rhs.x()));
    EXPECT_FLOAT_EQ(float(lhs.y()), float(rhs.y()));
    EXPECT_FLOAT_EQ(float(lhs.z()), float(rhs.z()));
  };

  Object &o = Object::Create();
  Camera &c = o.AddComponent<Camera>();

  expectV(c.backgroundColor(), Vec3r::zero());

  EXPECT_EQ(c.orthographic(), false);
  EXPECT_EQ(c.perspective(), true);
  EXPECT_FLOAT_EQ(c.fieldOfView(), 60_R);
  EXPECT_FLOAT_EQ(c.aspect(), 1_R);
  EXPECT_FLOAT_EQ(c.nearClipPlane(), 0.1_R);
  EXPECT_FLOAT_EQ(c.farClipPlane(), 100_R);

  c.backgroundColor() = Vec3r::one();
  expectV(c.backgroundColor(), Vec3r::one());

  c.orthographic() = true;
  EXPECT_EQ(c.orthographic(), true);
  EXPECT_EQ(c.perspective(), false);

  c.perspective() = true;
  EXPECT_EQ(c.orthographic(), false);
  EXPECT_EQ(c.perspective(), true);

  c.fieldOfView() = 45_R;
  EXPECT_FLOAT_EQ(c.fieldOfView(), 45_R);

  c.aspect() = 0.8_R;
  EXPECT_FLOAT_EQ(c.aspect(), 0.8_R);

  c.nearClipPlane() = 1_R;
  EXPECT_FLOAT_EQ(c.nearClipPlane(), 1_R);

  c.farClipPlane() = 1000_R;
  EXPECT_FLOAT_EQ(c.farClipPlane(), 1000_R);
}

} // namespace ARIA
