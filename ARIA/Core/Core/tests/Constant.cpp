#include "ARIA/Constant.h"

#include <gtest/gtest.h>

#include <array>

namespace ARIA {

namespace {

struct MyIntegralConstant0 {
  static constexpr int value = 0;
};

struct MyIntegralConstant1 {
  static const inline int value = 0;
};

struct MyFakeIntegralConstant {
  static inline int value = 0;
};

struct MyFloatingPointConstant0 {
  static constexpr float value = 0;
};

struct MyFloatingPointConstant1 {
  static const inline float value = 0;
};

struct MyFakeFloatingPointConstant {
  static inline float value = 0;
};

} // namespace

TEST(Constant, Base) {
  // Constant Arithmetic.
  static_assert(!ConstantArithmetic<int>);
  static_assert(!ConstantArithmetic<float>);

  static_assert(ConstantArithmetic<std::true_type>);
  static_assert(!ConstantArithmetic<std::true_type &>);
  static_assert(!ConstantArithmetic<std::false_type &&>);
  static_assert(ConstantArithmetic<std::integral_constant<int, 0>>);
  static_assert(ConstantArithmetic<C<2>>);
  static_assert(ConstantArithmetic<const C<2u>>);
  static_assert(ConstantArithmetic<const C<2ll>>);
  static_assert(ConstantArithmetic<C<2llu>>);
  static_assert(ConstantArithmetic<cute::C<2>>);
  static_assert(ConstantArithmetic<const cute::C<2u>>);
  static_assert(ConstantArithmetic<const cute::C<2ll>>);
  static_assert(ConstantArithmetic<cute::C<2llu>>);
  static_assert(ConstantArithmetic<cute::Int<2>>);
  static_assert(ConstantArithmetic<cute::_3>);

  static_assert(ConstantArithmetic<MyIntegralConstant0>);
  static_assert(ConstantArithmetic<const MyIntegralConstant0>);
  static_assert(!ConstantArithmetic<MyIntegralConstant0 &>);
  static_assert(!ConstantArithmetic<MyIntegralConstant0 &&>);
  static_assert(ConstantArithmetic<MyIntegralConstant1>);
  static_assert(!ConstantArithmetic<MyFakeIntegralConstant>);

  static_assert(ConstantArithmetic<MyFloatingPointConstant0>);
  static_assert(ConstantArithmetic<MyFloatingPointConstant1>);
  static_assert(!ConstantArithmetic<MyFakeFloatingPointConstant>);

  // Constant integral.
  static_assert(!ConstantIntegral<int>);
  static_assert(!ConstantIntegral<float>);

  static_assert(ConstantIntegral<std::true_type>);
  static_assert(!ConstantIntegral<std::true_type &>);
  static_assert(!ConstantIntegral<std::false_type &&>);
  static_assert(ConstantIntegral<std::integral_constant<int, 0>>);
  static_assert(ConstantIntegral<C<2>>);
  static_assert(ConstantIntegral<const C<2u>>);
  static_assert(ConstantIntegral<const C<2ll>>);
  static_assert(ConstantIntegral<C<2llu>>);
  static_assert(ConstantIntegral<cute::C<2>>);
  static_assert(ConstantIntegral<const cute::C<2u>>);
  static_assert(ConstantIntegral<const cute::C<2ll>>);
  static_assert(ConstantIntegral<cute::C<2llu>>);
  static_assert(ConstantIntegral<cute::Int<2>>);
  static_assert(ConstantIntegral<cute::_3>);

  static_assert(ConstantIntegral<MyIntegralConstant0>);
  static_assert(ConstantIntegral<const MyIntegralConstant0>);
  static_assert(!ConstantIntegral<MyIntegralConstant0 &>);
  static_assert(!ConstantIntegral<MyIntegralConstant0 &&>);
  static_assert(ConstantIntegral<MyIntegralConstant1>);
  static_assert(!ConstantIntegral<MyFakeIntegralConstant>);

  static_assert(!ConstantIntegral<MyFloatingPointConstant0>);
  static_assert(!ConstantIntegral<MyFloatingPointConstant1>);
  static_assert(!ConstantIntegral<MyFakeFloatingPointConstant>);
}

} // namespace ARIA
