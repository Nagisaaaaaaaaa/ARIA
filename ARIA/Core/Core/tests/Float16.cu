#include "ARIA/Float16.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(Float16, Base) {
  float16 a{0.1F};
  float16 b{0.2F};
  float16 c = a + b;
  EXPECT_TRUE(std::abs(static_cast<float>(c) - 0.3F) < 0.0001F);

  EXPECT_TRUE(static_cast<float>(std::numeric_limits<float16>::infinity()) == std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-std::numeric_limits<float16>::infinity()) == -std::numeric_limits<float>::infinity());

  EXPECT_TRUE(static_cast<float>(cuda::std::numeric_limits<float16>::infinity()) ==
              cuda::std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-cuda::std::numeric_limits<float16>::infinity()) ==
              -cuda::std::numeric_limits<float>::infinity());
}

} // namespace ARIA
