#include "ARIA/BFloat16.h"

#include <cuda/api.hpp>
#include <gtest/gtest.h>

namespace ARIA {

namespace {

ARIA_KERNEL void TestCUDAKernel() {
  bfloat16 a{0.1F};
  bfloat16 b{0.2F};
  bfloat16 c = a + b;
  ARIA_ASSERT(std::abs(static_cast<float>(c) - 0.3F) < 0.001F);

  ARIA_ASSERT(static_cast<float>(std::numeric_limits<bfloat16>::infinity()) == std::numeric_limits<float>::infinity());
  ARIA_ASSERT(static_cast<float>(-std::numeric_limits<bfloat16>::infinity()) ==
              -std::numeric_limits<float>::infinity());

  ARIA_ASSERT(static_cast<float>(cuda::std::numeric_limits<bfloat16>::infinity()) ==
              cuda::std::numeric_limits<float>::infinity());
  ARIA_ASSERT(static_cast<float>(-cuda::std::numeric_limits<bfloat16>::infinity()) ==
              -cuda::std::numeric_limits<float>::infinity());
}

void TestCUDA() {
  try {
    TestCUDAKernel<<<1, 1>>>();
    cuda::device::current::get().synchronize();
  } catch (...) { EXPECT_FALSE(true); }
}

} // namespace

TEST(BFloat16, Base) {
  bfloat16 a{0.1F};
  bfloat16 b{0.2F};
  bfloat16 c = a + b;
  EXPECT_TRUE(std::abs(static_cast<float>(c) - 0.3F) < 0.001F);

  EXPECT_TRUE(static_cast<float>(std::numeric_limits<bfloat16>::infinity()) == std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-std::numeric_limits<bfloat16>::infinity()) ==
              -std::numeric_limits<float>::infinity());

  EXPECT_TRUE(static_cast<float>(cuda::std::numeric_limits<bfloat16>::infinity()) ==
              cuda::std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-cuda::std::numeric_limits<bfloat16>::infinity()) ==
              -cuda::std::numeric_limits<float>::infinity());

  TestCUDA();
}

} // namespace ARIA
