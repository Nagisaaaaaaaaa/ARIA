#include "ARIA/Float16.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(Float16, Base) {
  EXPECT_TRUE(static_cast<float>(std::numeric_limits<float16>::infinity()) == std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-std::numeric_limits<float16>::infinity()) == -std::numeric_limits<float>::infinity());

  EXPECT_TRUE(static_cast<float>(cuda::std::numeric_limits<float16>::infinity()) ==
              cuda::std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-cuda::std::numeric_limits<float16>::infinity()) ==
              -cuda::std::numeric_limits<float>::infinity());
}

} // namespace ARIA
