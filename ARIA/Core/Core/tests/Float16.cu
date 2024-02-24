#include "ARIA/Float16.h"

#include <cuda/api.hpp>
#include <gtest/gtest.h>

namespace ARIA {

namespace {

ARIA_KERNEL void TestBaseCUDAKernel() {
  float16 a{0.1F};
  float16 b{0.2F};
  float16 c = a + b;
  ARIA_ASSERT(std::abs(static_cast<float>(c) - 0.3F) < 0.00025F);

  ARIA_ASSERT(static_cast<float>(std::numeric_limits<float16>::infinity()) == std::numeric_limits<float>::infinity());
  ARIA_ASSERT(static_cast<float>(-std::numeric_limits<float16>::infinity()) == -std::numeric_limits<float>::infinity());

  ARIA_ASSERT(static_cast<float>(cuda::std::numeric_limits<float16>::infinity()) ==
              cuda::std::numeric_limits<float>::infinity());
  ARIA_ASSERT(static_cast<float>(-cuda::std::numeric_limits<float16>::infinity()) ==
              -cuda::std::numeric_limits<float>::infinity());
}

ARIA_KERNEL void TestMathCUDAKernel() {
  float16 a{0.1F};
  float16 b{-0.1F};

  ARIA_ASSERT(abs(a) == a);
  ARIA_ASSERT(abs(b) == a);
  ARIA_ASSERT(std::abs(a) == a);
  ARIA_ASSERT(std::abs(b) == a);
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

TEST(Float16, Base) {
  float16 a{0.1F};
  float16 b{0.2F};
  float16 c = a + b;
  EXPECT_TRUE(std::abs(static_cast<float>(c) - 0.3F) < 0.00025F);

  EXPECT_TRUE(static_cast<float>(std::numeric_limits<float16>::infinity()) == std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-std::numeric_limits<float16>::infinity()) == -std::numeric_limits<float>::infinity());

  EXPECT_TRUE(static_cast<float>(cuda::std::numeric_limits<float16>::infinity()) ==
              cuda::std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-cuda::std::numeric_limits<float16>::infinity()) ==
              -cuda::std::numeric_limits<float>::infinity());

  TestBaseCUDA();
}

TEST(Float16, Math) {
  float16 a{0.1F};
  float16 b{-0.1F};

  EXPECT_EQ(abs(a), a);
  EXPECT_EQ(abs(b), a);
  EXPECT_EQ(std::abs(a), a);
  EXPECT_EQ(std::abs(b), a);

  TestMathCUDA();
}

} // namespace ARIA
