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
  ARIA_ASSERT(std::numeric_limits<bfloat16>::quiet_NaN() != std::numeric_limits<bfloat16>::quiet_NaN());

  ARIA_ASSERT(cuda::std::numeric_limits<bfloat16>::min() > bfloat16{});
  ARIA_ASSERT(cuda::std::numeric_limits<bfloat16>::max() > bfloat16{});
  ARIA_ASSERT(static_cast<float>(cuda::std::numeric_limits<bfloat16>::infinity()) ==
              cuda::std::numeric_limits<float>::infinity());
  ARIA_ASSERT(static_cast<float>(-cuda::std::numeric_limits<bfloat16>::infinity()) ==
              -cuda::std::numeric_limits<float>::infinity());
  ARIA_ASSERT(cuda::std::numeric_limits<bfloat16>::quiet_NaN() != cuda::std::numeric_limits<bfloat16>::quiet_NaN());
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

  ARIA_ASSERT(max(a, b) == a);
  ARIA_ASSERT(min(a, b) == b);
  ARIA_ASSERT(std::max(a, b) == a);
  ARIA_ASSERT(std::min(a, b) == b);
  ARIA_ASSERT(cuda::std::max(a, b) == a);
  ARIA_ASSERT(cuda::std::min(a, b) == b);
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
  EXPECT_NE(std::numeric_limits<bfloat16>::quiet_NaN(), std::numeric_limits<bfloat16>::quiet_NaN());

  EXPECT_TRUE(cuda::std::numeric_limits<bfloat16>::min() > bfloat16{});
  EXPECT_TRUE(cuda::std::numeric_limits<bfloat16>::max() > bfloat16{});
  EXPECT_TRUE(static_cast<float>(cuda::std::numeric_limits<bfloat16>::infinity()) ==
              cuda::std::numeric_limits<float>::infinity());
  EXPECT_TRUE(static_cast<float>(-cuda::std::numeric_limits<bfloat16>::infinity()) ==
              -cuda::std::numeric_limits<float>::infinity());
  EXPECT_NE(cuda::std::numeric_limits<bfloat16>::quiet_NaN(), cuda::std::numeric_limits<bfloat16>::quiet_NaN());

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

  EXPECT_EQ(max(a, b), a);
  EXPECT_EQ(min(a, b), b);
  EXPECT_EQ(std::max(a, b), a);
  EXPECT_EQ(std::min(a, b), b);
  EXPECT_EQ(cuda::std::max(a, b), a);
  EXPECT_EQ(cuda::std::min(a, b), b);

  TestMathCUDA();
}

TEST(BFloat16, Python) {
  Python::ScopedInterpreter guard{};

  Python::Module main = guard.Import("__main__");
  Python::Dict local{main};

  bfloat16 a{0.1F};
  bfloat16 b{0.2F};
  bfloat16 a_add_b = a + b;
  bfloat16 a_sub_b = a - b;
  bfloat16 a_mul_b = a * b;
  bfloat16 a_div_b = a / b;

  local["a_copy"] = a;
  local["minusA_copy"] = -a;
  local["a_add_b"] = a_add_b;
  local["a_sub_b"] = &a_sub_b;
  local["a_mul_b"] = a_mul_b;
  local["a_div_b"] = &a_div_b;

  try {
    py::exec("a = bfloat16(0.1)\n"
             "b = bfloat16(0.2)\n"
             "c = bfloat16(233)\n"
             "\n"
             "assert a < b\n"
             "assert b > a\n"
             "assert a <= b\n"
             "assert b >= a\n"
             "assert +a == a_copy\n"
             "assert -a == minusA_copy\n"
             "assert a + b == a_add_b\n"
             "assert a - b == a_sub_b\n"
             "assert a * b == a_mul_b\n"
             "assert a / b == a_div_b\n"
             "\n"
             "assert abs(-a) == a\n"
             "assert max(a, b) == b\n"
             "assert min(a, b) == a\n",
             py::globals(), local);
  } catch (std::exception &e) {
    fmt::print("{}\n", e.what());
    EXPECT_FALSE(true);
  }
}

} // namespace ARIA
