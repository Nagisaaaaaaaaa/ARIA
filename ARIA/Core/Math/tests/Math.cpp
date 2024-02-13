#include "ARIA/Math.h"
#include "ARIA/Vec.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(Math, Base) {
  // Constants.
  EXPECT_FLOAT_EQ(pi<float>, 3.14159265F);
  EXPECT_FLOAT_EQ(piInv<float>, 1.0F / 3.14159265F);
  EXPECT_FLOAT_EQ(e<float>, 2.7182818284F);

  // Deg2Rad.
  EXPECT_FLOAT_EQ(180.0F * deg2Rad<float>, pi<float>);
  {
    Vec3f deg{180.0F, 90.0F, 45.0F};
    Vec3f rad = deg * deg2Rad<float>;
    EXPECT_FLOAT_EQ(rad.x(), pi<float>);
    EXPECT_FLOAT_EQ(rad.y(), pi<float> / 2);
    EXPECT_FLOAT_EQ(rad.z(), pi<float> / 4);
  }

  // Rad2Deg.
  EXPECT_FLOAT_EQ(pi<float> * rad2Deg<float>, 180.0F);
  {
    Vec3f rad = {pi<float>, pi<float> / 2, pi<float> / 4};
    Vec3f deg = rad * rad2Deg<float>;
    EXPECT_FLOAT_EQ(deg.x(), 180.0F);
    EXPECT_FLOAT_EQ(deg.y(), 90.0F);
    EXPECT_FLOAT_EQ(deg.z(), 45.0F);
  }

  // Lerp.
  EXPECT_FLOAT_EQ(Lerp(1.1F, 2.2F, 0.25F), 1.375F);
  {
    Vec3f x = {1.1F, 0.11F, 0.011F};
    Vec3f y = {2.2F, 0.22F, 0.022F};
    Vec3f z = Lerp(x, y, 0.25F);
    EXPECT_FLOAT_EQ(z.x(), 1.375F);
    EXPECT_FLOAT_EQ(z.y(), 0.1375F);
    EXPECT_FLOAT_EQ(z.z(), 0.01375F);
  }
}

} // namespace ARIA
