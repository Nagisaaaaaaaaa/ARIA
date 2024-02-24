#include "ARIA/BFloat16.h"

#include <cuda/api.hpp>
#include <gtest/gtest.h>

namespace ARIA {

namespace {

ARIA_KERNEL void TestBaseCUDAKernel() {
  ARIA_ASSERT(bfloat16{} == CUDART_ZERO_BF16);

  bfloat16 a{0.1F};
  bfloat16 b{0.2F};
  bfloat16 c = a + b;
  ARIA_ASSERT(std::abs(static_cast<float>(c) - 0.3F) < 0.001F);

  ARIA_ASSERT(std::numeric_limits<bfloat16>::min() > bfloat16{});
  ARIA_ASSERT(std::numeric_limits<bfloat16>::max() > bfloat16{});
  ARIA_ASSERT(static_cast<float>(std::numeric_limits<bfloat16>::infinity()) == std::numeric_limits<float>::infinity());
  ARIA_ASSERT(static_cast<float>(-std::numeric_limits<bfloat16>::infinity()) ==
              -std::numeric_limits<float>::infinity());

  ARIA_ASSERT(cuda::std::numeric_limits<bfloat16>::min() > bfloat16{});
  ARIA_ASSERT(cuda::std::numeric_limits<bfloat16>::max() > bfloat16{});
  ARIA_ASSERT(static_cast<float>(cuda::std::numeric_limits<bfloat16>::infinity()) ==
              cuda::std::numeric_limits<float>::infinity());
  ARIA_ASSERT(static_cast<float>(-cuda::std::numeric_limits<bfloat16>::infinity()) ==
              -cuda::std::numeric_limits<float>::infinity());
}

ARIA_KERNEL void TestMathCUDAKernel() {
  bfloat16 a{0.1F};
  bfloat16 b{-0.1F};

  ARIA_ASSERT(abs(a) == a);
  ARIA_ASSERT(abs(b) == a);
  ARIA_ASSERT(std::abs(a) == a);
  ARIA_ASSERT(std::abs(b) == a);
  ARIA_ASSERT(cuda::std::abs(a) == a);
  ARIA_ASSERT(cuda::std::abs(b) == a);
}

void TestBaseCUDA() {
  try {
    TestBaseCUDAKernel<<<1, 1>>>();
    cuda::device::current::get().synchronize();
  } catch (...) { EXPECT_FALSE(true); }
}

void TestMathCUDA() {
  try {
    TestMathCUDAKernel<<<1, 1>>>();
    cuda::device::current::get().synchronize();
  } catch (...) { EXPECT_FALSE(true); }
}

} // namespace

TEST(BFloat16, Base) {
  EXPECT_EQ(bfloat16{}, CUDART_ZERO_BF16);

  bfloat16 a{0.1F};
  bfloat16 b{0.2F};
  bfloat16 c = a + b;
  EXPECT_TRUE(std::abs(static_cast<float>(c) - 0.3F) < 0.001F);

  EXPECT_TRUE(std::numeric_limits<bfloat16>::min() > bfloat16{});
  EXPECT_TRUE(std::numeric_limits<bfloat16>::max() > bfloat16{});
  EXPECT_TRUE(static_cast<float>(std::numeric_limits<bfloat16>::infinity()) == std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-std::numeric_limits<bfloat16>::infinity()) ==
              -std::numeric_limits<float>::infinity());

  EXPECT_TRUE(cuda::std::numeric_limits<bfloat16>::min() > bfloat16{});
  EXPECT_TRUE(cuda::std::numeric_limits<bfloat16>::max() > bfloat16{});
  EXPECT_TRUE(static_cast<float>(cuda::std::numeric_limits<bfloat16>::infinity()) ==
              cuda::std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-cuda::std::numeric_limits<bfloat16>::infinity()) ==
              -cuda::std::numeric_limits<float>::infinity());

  TestBaseCUDA();
}

TEST(BFloat16, Math) {
  bfloat16 a{0.1F};
  bfloat16 b{-0.1F};

  EXPECT_EQ(abs(a), a);
  EXPECT_EQ(abs(b), a);
  EXPECT_EQ(std::abs(a), a);
  EXPECT_EQ(std::abs(b), a);
  EXPECT_EQ(cuda::std::abs(a), a);
  EXPECT_EQ(cuda::std::abs(b), a);

  TestMathCUDA();
}

} // namespace ARIA
