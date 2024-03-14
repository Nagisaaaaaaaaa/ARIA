#pragma once

#if defined(ARIA_USE_32BIT_REAL)
  #define ASSERT_REAL_EQ(x, y) ASSERT_FLOAT_EQ(x, y)
  #define EXPECT_REAL_EQ(x, y) EXPECT_FLOAT_EQ(x, y)
#else
  #define ASSERT_REAL_EQ(x, y) ASSERT_DOUBLE_EQ(x, y)
  #define EXPECT_REAL_EQ(x, y) EXPECT_DOUBLE_EQ(x, y)
#endif
