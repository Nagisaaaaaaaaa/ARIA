#include "ARIA/Layout.h"
#include "ARIA/Let.h"

#include <gtest/gtest.h>

#include <queue>

namespace ARIA {

using cute::_0;
using cute::_1;
using cute::_2;
using cute::_3;
using cute::_4;

TEST(Layout, Base) {
  // Arithmetic type.
  static_assert(std::is_same_v<layout::detail::arithmetic_domain_t<int>, int>);
  static_assert(std::is_same_v<layout::detail::arithmetic_domain_t<const int>, int>);
  static_assert(std::is_same_v<layout::detail::arithmetic_domain_t<const int &>, int>);
  static_assert(std::is_same_v<layout::detail::arithmetic_domain_t<C<1>>, int>);
  static_assert(std::is_same_v<layout::detail::arithmetic_domain_t<const C<1>>, int>);
  static_assert(std::is_same_v<layout::detail::arithmetic_domain_t<const C<1> &>, int>);
  static_assert(std::is_same_v<layout::detail::arithmetic_domain_t<std::string>, void>);

  // Make tup.
  {
    Tup v{1, 2.0F, Tup{3.0, std::string{"4"}}};
    let vSub = Tup{3.0, std::string{"4"}};
    EXPECT_EQ(get<0>(v), 1);
    EXPECT_EQ(get<1>(v), 2.0F);
    EXPECT_EQ(get<2>(v), vSub);
    EXPECT_EQ(get<0>(get<2>(v)), 3.0);
    EXPECT_EQ(get<1>(get<2>(v)), std::string{"4"});
  }

  // Make crd.
  {
    Crd v{1, 2.0F, C<3U>{}, C<4.0>{}};
    EXPECT_EQ(get<0>(v), 1);
    EXPECT_EQ(get<1>(v), 2.0F);
    static_assert(get<2>(v) == C<3U>{});
    static_assert(get<3>(v) == C<4.0>{});
  }

  static_assert(rank(Crd{}) == 0);
  static_assert(rank(Crd{0}) == 1);
  static_assert(rank(Crd{_0{}}) == 1);
  static_assert(rank(Crd{0, 1}) == 2);
  static_assert(rank(Crd{_0{}, 1}) == 2);
  static_assert(rank(Crd{0, _1{}}) == 2);
  static_assert(rank(Crd{_0{}, _1{}}) == 2);

  static_assert(is_static_v<decltype(Crd{})>);
  static_assert(!is_static_v<decltype(Crd{0})>);
  static_assert(is_static_v<decltype(Crd{_0{}})>);
  static_assert(!is_static_v<decltype(Crd{0, 1})>);
  static_assert(!is_static_v<decltype(Crd{_0{}, 1})>);
  static_assert(!is_static_v<decltype(Crd{0, _1{}})>);
  static_assert(is_static_v<decltype(Crd{_0{}, _1{}})>);

  // Make shape.
  static_assert(is_static_v<decltype(make_shape(_2{}, make_shape(_2{}, _2{})))>);
  static_assert(!is_static_v<decltype(make_shape(2, make_shape(_2{}, _2{})))>);
  static_assert(!is_static_v<decltype(make_shape(_2{}, make_shape(2, _2{})))>);
  static_assert(!is_static_v<decltype(make_shape(_2{}, make_shape(_2{}, 2)))>);
  static_assert(is_static_v<decltype(make_shape(_2{}, make_shape(_2{}, _2{})))>);
  static_assert(is_static_v<decltype(make_shape(_2{}, make_shape(_2{}, _2{})))>);
  static_assert(is_static_v<decltype(make_shape(_2{}, make_shape(_2{}, _2{})))>);
  static_assert(!is_static_v<decltype(make_shape(2, make_shape(2, 2)))>);

  // Make stride.
  static_assert(is_static_v<decltype(make_stride(_4{}, make_stride(_2{}, _1{})))>);
  static_assert(is_static_v<decltype(make_stride(_4{}, make_stride(_2{}, _1{})))>);
  static_assert(is_static_v<decltype(make_stride(_4{}, make_stride(_2{}, _1{})))>);
  static_assert(is_static_v<decltype(make_stride(_4{}, make_stride(_2{}, _1{})))>);
  static_assert(!is_static_v<decltype(make_stride(4, make_stride(_2{}, _1{})))>);
  static_assert(!is_static_v<decltype(make_stride(_4{}, make_stride(2, _1{})))>);
  static_assert(!is_static_v<decltype(make_stride(_4{}, make_stride(_2{}, 1)))>);
  static_assert(!is_static_v<decltype(make_stride(4, make_stride(2, 1)))>);

  // Make layout.
  static_assert(is_static_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                 make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_static_v<decltype(make_layout(make_shape(2, make_shape(_2{}, _2{})),
                                                  make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_static_v<decltype(make_layout(make_shape(_2{}, make_shape(2, _2{})),
                                                  make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_static_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, 2)),
                                                  make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_static_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                  make_stride(4, make_stride(_2{}, _1{}))))>);
  static_assert(!is_static_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                  make_stride(_4{}, make_stride(2, _1{}))))>);
  static_assert(!is_static_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                  make_stride(_4{}, make_stride(_2{}, 1))))>);
  static_assert(
      !is_static_v<decltype(make_layout(make_shape(2, make_shape(2, 2)), make_stride(4, make_stride(2, 1))))>);

  // Make layout major.
  static_assert(std::is_same_v<decltype(make_layout(make_shape(1, 2))), decltype(make_layout_major<LayoutLeft>(3, 4))>);
  static_assert(
      !std::is_same_v<decltype(make_layout(make_shape(1, 2))), decltype(make_layout_major<LayoutRight>(3, 4))>);
  static_assert(std::is_same_v<decltype(make_layout(make_shape(1, 2))), decltype(make_layout_major(3, 4))>);
}

TEST(Layout, SizeAndCosize) {
  auto testDynamic1 = [&](auto x) {
    auto layout = make_layout_major(x);

    EXPECT_EQ(size(layout), 0);
    EXPECT_EQ(cosize_safe(layout), 0);
  };

  auto testDynamic2 = [&](auto x, auto y) {
    auto layout = make_layout_major(x, y);

    EXPECT_EQ(size(layout), 0);
    EXPECT_EQ(cosize_safe(layout), 0);
  };

  auto testDynamic3 = [&](auto x, auto y, auto z) {
    auto layout = make_layout_major(x, y, z);

    EXPECT_EQ(size(layout), 0);
    EXPECT_EQ(cosize_safe(layout), 0);
  };

  auto testStatic1 = [&](auto x) {
    auto layout = make_layout_major(x);

    EXPECT_EQ(size(layout), 0);
    EXPECT_EQ(cosize_safe(layout), 0);

    static_assert(size(layout) == 0);
    static_assert(cosize_safe(layout) == 0);
    static_assert(cosize_safe_v<decltype(layout)> == 0);
  };

  auto testStatic2 = [&](auto x, auto y) {
    auto layout = make_layout_major(x, y);

    EXPECT_EQ(size(layout), 0);
    EXPECT_EQ(cosize_safe(layout), 0);

    static_assert(size(layout) == 0);
    static_assert(cosize_safe(layout) == 0);
    static_assert(cosize_safe_v<decltype(layout)> == 0);
  };

  auto testStatic3 = [&](auto x, auto y, auto z) {
    auto layout = make_layout_major(x, y, z);

    EXPECT_EQ(size(layout), 0);
    EXPECT_EQ(cosize_safe(layout), 0);

    static_assert(size(layout) == 0);
    static_assert(cosize_safe(layout) == 0);
    static_assert(cosize_safe_v<decltype(layout)> == 0);
  };

  // All zeros.
  testDynamic1(0);

  testDynamic2(0, 0);
  testDynamic2(1, 0);
  testDynamic2(2, 0);
  testDynamic2(0, 1);
  testDynamic2(0, 2);

  testDynamic3(0, 0, 0);
  testDynamic3(1, 0, 0);
  testDynamic3(2, 0, 0);
  testDynamic3(0, 1, 0);
  testDynamic3(0, 2, 0);
  testDynamic3(0, 0, 1);
  testDynamic3(0, 0, 2);
  testDynamic3(1, 1, 0);
  testDynamic3(2, 2, 0);
  testDynamic3(0, 1, 1);
  testDynamic3(0, 2, 2);
  testDynamic3(1, 0, 1);
  testDynamic3(2, 0, 2);

  testStatic1(_0{});

  testStatic2(_0{}, _0{});
  testStatic2(_1{}, _0{});
  testStatic2(_2{}, _0{});
  testStatic2(_0{}, _1{});
  testStatic2(_0{}, _2{});

  testStatic3(_0{}, _0{}, _0{});
  testStatic3(_1{}, _0{}, _0{});
  testStatic3(_2{}, _0{}, _0{});
  testStatic3(_0{}, _1{}, _0{});
  testStatic3(_0{}, _2{}, _0{});
  testStatic3(_0{}, _0{}, _1{});
  testStatic3(_0{}, _0{}, _2{});
  testStatic3(_1{}, _1{}, _0{});
  testStatic3(_2{}, _2{}, _0{});
  testStatic3(_0{}, _1{}, _1{});
  testStatic3(_0{}, _2{}, _2{});
  testStatic3(_1{}, _0{}, _1{});
  testStatic3(_2{}, _0{}, _2{});
}

TEST(Layout, Is) {
  // Is layout const size.
  static_assert(is_layout_const_size_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                            make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_layout_const_size_v<decltype(make_layout(make_shape(2, make_shape(_2{}, _2{})),
                                                             make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_layout_const_size_v<decltype(make_layout(make_shape(_2{}, make_shape(2, _2{})),
                                                             make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_layout_const_size_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, 2)),
                                                             make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(is_layout_const_size_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                            make_stride(4, make_stride(_2{}, _1{}))))>);
  static_assert(is_layout_const_size_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                            make_stride(_4{}, make_stride(2, _1{}))))>);
  static_assert(is_layout_const_size_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                            make_stride(_4{}, make_stride(_2{}, 1))))>);
  static_assert(!is_layout_const_size_v<decltype(make_layout(make_shape(2, make_shape(2, 2)),
                                                             make_stride(4, make_stride(2, 1))))>);

  // Is layout const size at.
  static_assert(is_layout_const_size_at_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                               make_stride(_4{}, make_stride(_2{}, _1{})))),
                                          0>);
  static_assert(is_layout_const_size_at_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                               make_stride(_4{}, make_stride(_2{}, _1{})))),
                                          1>);
  static_assert(
      !is_layout_const_size_at_v<
          decltype(make_layout(make_shape(2, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})))), 0>);
  static_assert(
      is_layout_const_size_at_v<
          decltype(make_layout(make_shape(2, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})))), 1>);
  static_assert(
      is_layout_const_size_at_v<
          decltype(make_layout(make_shape(_2{}, make_shape(2, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})))), 0>);
  static_assert(
      !is_layout_const_size_at_v<
          decltype(make_layout(make_shape(_2{}, make_shape(2, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})))), 1>);
  static_assert(
      is_layout_const_size_at_v<
          decltype(make_layout(make_shape(_2{}, make_shape(_2{}, 2)), make_stride(_4{}, make_stride(_2{}, _1{})))), 0>);
  static_assert(
      !is_layout_const_size_at_v<
          decltype(make_layout(make_shape(_2{}, make_shape(_2{}, 2)), make_stride(_4{}, make_stride(_2{}, _1{})))), 1>);
  static_assert(
      is_layout_const_size_at_v<
          decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(4, make_stride(_2{}, _1{})))), 0>);
  static_assert(
      is_layout_const_size_at_v<
          decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(4, make_stride(_2{}, _1{})))), 1>);
  static_assert(
      is_layout_const_size_at_v<
          decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(2, _1{})))), 0>);
  static_assert(
      is_layout_const_size_at_v<
          decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(2, _1{})))), 1>);
  static_assert(
      is_layout_const_size_at_v<
          decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, 1)))), 0>);
  static_assert(
      is_layout_const_size_at_v<
          decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, 1)))), 1>);
  static_assert(!is_layout_const_size_at_v<
                decltype(make_layout(make_shape(2, make_shape(2, 2)), make_stride(4, make_stride(2, 1)))), 0>);
  static_assert(!is_layout_const_size_at_v<
                decltype(make_layout(make_shape(2, make_shape(2, 2)), make_stride(4, make_stride(2, 1)))), 1>);

  // Is layout const cosize safe.
  static_assert(is_layout_const_cosize_safe_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                                   make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_layout_const_cosize_safe_v<decltype(make_layout(make_shape(2, make_shape(_2{}, _2{})),
                                                                    make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_layout_const_cosize_safe_v<decltype(make_layout(make_shape(_2{}, make_shape(2, _2{})),
                                                                    make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_layout_const_cosize_safe_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, 2)),
                                                                    make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(!is_layout_const_cosize_safe_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                                    make_stride(4, make_stride(_2{}, _1{}))))>);
  static_assert(!is_layout_const_cosize_safe_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                                    make_stride(_4{}, make_stride(2, _1{}))))>);
  static_assert(!is_layout_const_cosize_safe_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                                    make_stride(_4{}, make_stride(_2{}, 1))))>);
  static_assert(!is_layout_const_cosize_safe_v<decltype(make_layout(make_shape(2, make_shape(2, 2)),
                                                                    make_stride(4, make_stride(2, 1))))>);

  // Is co-layout.
  static_assert(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})))));
  static_assert(is_co_layout_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                    make_stride(_4{}, make_stride(_2{}, _1{}))))>);
  static_assert(CoLayout<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                              make_stride(_4{}, make_stride(_2{}, _1{}))))>);

  static_assert(
      !is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _2{})))));
  static_assert(!is_co_layout_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                                     make_stride(_4{}, make_stride(_2{}, _2{}))))>);
  static_assert(!CoLayout<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
                                               make_stride(_4{}, make_stride(_2{}, _2{}))))>);

  EXPECT_TRUE(
      is_co_layout(make_layout(make_shape(2, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})))));
  EXPECT_TRUE(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(2, _2{})), make_stride(_4{}, make_stride(_2{}, _1{})))));
  EXPECT_TRUE(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, 2)), make_stride(_4{}, make_stride(_2{}, _1{})))));
  EXPECT_TRUE(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(4, make_stride(_2{}, _1{})))));
  EXPECT_TRUE(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(2, _1{})))));
  EXPECT_TRUE(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, 1)))));
  EXPECT_TRUE(is_co_layout(make_layout(make_shape(2, make_shape(2, 2)), make_stride(4, make_stride(2, 1)))));

  EXPECT_FALSE(
      is_co_layout(make_layout(make_shape(2, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, _2{})))));
  EXPECT_FALSE(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(2, _2{})), make_stride(_4{}, make_stride(_2{}, _2{})))));
  EXPECT_FALSE(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, 2)), make_stride(_4{}, make_stride(_2{}, _2{})))));
  EXPECT_FALSE(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(4, make_stride(_2{}, _2{})))));
  EXPECT_FALSE(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(2, _2{})))));
  EXPECT_FALSE(
      is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})), make_stride(_4{}, make_stride(_2{}, 2)))));
  EXPECT_FALSE(is_co_layout(make_layout(make_shape(2, make_shape(2, 2)), make_stride(4, make_stride(2, 2)))));
}

TEST(Layout, Cast) {
  // To `std::array`.
  {
    static_assert(ToArray(Crd{0}) == std::array{0});
    static_assert(ToArray(Crd{0_I}) == std::array{0});

    static_assert(ToArray(Crd{0U}) == std::array{0U});
    static_assert(ToArray(Crd{0_U}) == std::array{0U});

    static_assert(ToArray(Crd{0, 1}) == std::array{0, 1});
    static_assert(ToArray(Crd{0_I, 1}) == std::array{0, 1});
    static_assert(ToArray(Crd{0, 1_I}) == std::array{0, 1});
    static_assert(ToArray(Crd{0_I, 1_I}) == std::array{0, 1});

    static_assert(ToArray(Crd{0U, 1U}) == std::array{0U, 1U});
    static_assert(ToArray(Crd{0_U, 1U}) == std::array{0U, 1U});
    static_assert(ToArray(Crd{0U, 1_U}) == std::array{0U, 1U});
    static_assert(ToArray(Crd{0_U, 1_U}) == std::array{0U, 1U});

    static_assert(ToArray(Crd{0, 1, 2}) == std::array{0, 1, 2});
    static_assert(ToArray(Crd{0_I, 1, 2}) == std::array{0, 1, 2});
    static_assert(ToArray(Crd{0, 1_I, 2}) == std::array{0, 1, 2});
    static_assert(ToArray(Crd{0_I, 1_I, 2}) == std::array{0, 1, 2});
    static_assert(ToArray(Crd{0, 1, 2_I}) == std::array{0, 1, 2});
    static_assert(ToArray(Crd{0_I, 1, 2_I}) == std::array{0, 1, 2});
    static_assert(ToArray(Crd{0, 1_I, 2_I}) == std::array{0, 1, 2});
    static_assert(ToArray(Crd{0_I, 1_I, 2_I}) == std::array{0, 1, 2});

    static_assert(ToArray(Crd{0U, 1U, 2U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Crd{0_U, 1U, 2U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Crd{0U, 1_U, 2U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Crd{0_U, 1_U, 2U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Crd{0U, 1U, 2_U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Crd{0_U, 1U, 2_U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Crd{0U, 1_U, 2_U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Crd{0_U, 1_U, 2_U}) == std::array{0U, 1U, 2U});
  }
}

TEST(Layout, OperatorsInt) {
  using Crd2 = Crd<int, int>;
  using Crd3 = Crd<int, int, int>;

  auto expectCrd2 = [](const Crd2 &crd, int x, int y) {
    EXPECT_EQ(get<0>(crd), x);
    EXPECT_EQ(get<1>(crd), y);
  };

  auto expectCrd3 = [](const Crd3 &crd, int x, int y, int z) {
    EXPECT_EQ(get<0>(crd), x);
    EXPECT_EQ(get<1>(crd), y);
    EXPECT_EQ(get<2>(crd), z);
  };

  {
    Crd2 a = cute::aria::layout::detail::FillCoords<int, int>(233);
    Crd3 b = cute::aria::layout::detail::FillCoords<int, int, int>(233);
    constexpr let c = cute::aria::layout::detail::FillCoords<C<233>, C<233>>(C<233>{});
    constexpr let d = cute::aria::layout::detail::FillCoords<C<233>, C<233>, C<233>>(C<233>{});
    expectCrd2(a, 233, 233);
    expectCrd3(b, 233, 233, 233);
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(Crd{233_I, 233_I})>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(Crd{233_I, 233_I, 233_I})>>);
  }

  // 2D.
  {
    Crd2 a{2, 7};
    Crd2 b{5, 11};
    Crd2 c = a + b;
    Crd2 d = a - b;
    Crd2 e = a * b;
    expectCrd2(c, 7, 18);
    expectCrd2(d, -3, -4);
    expectCrd2(e, 10, 77);
  }

  {
    constexpr let a = Crd{2_I, 7_I};
    constexpr let b = Crd{5_I, 11_I};
    constexpr let c = a + b;
    constexpr let d = a - b;
    constexpr let e = a * b;
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(Crd{7_I, 18_I})>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(Crd{-3_I, -4_I})>>);
    static_assert(std::is_same_v<decltype(e), std::add_const_t<decltype(Crd{10_I, 77_I})>>);
  }

  {
    let a = Crd{2_I, 7};
    let b = Crd{5_I, 11_I};
    let c = a + b;
    let d = a - b;
    let e = a * b;
    static_assert(std::is_same_v<decltype(c), decltype(Crd{7_I, 18})>);
    static_assert(std::is_same_v<decltype(d), decltype(Crd{-3_I, -4})>);
    static_assert(std::is_same_v<decltype(e), decltype(Crd{10_I, 77})>);
    EXPECT_EQ(get<1>(c), 18);
    EXPECT_EQ(get<1>(d), -4);
    EXPECT_EQ(get<1>(e), 77);
  }

  {
    Crd2 a{2, 7};
    int b = 5;
    Crd2 c0 = a + b;
    Crd2 c1 = a - b;
    Crd2 c2 = a * b;
    Crd2 c3 = b + a;
    Crd2 c4 = b - a;
    Crd2 c5 = b * a;
    expectCrd2(c0, 7, 12);
    expectCrd2(c1, -3, 2);
    expectCrd2(c2, 10, 35);
    expectCrd2(c3, 7, 12);
    expectCrd2(c4, 3, -2);
    expectCrd2(c5, 10, 35);
  }

  {
    constexpr let a = Crd{2_I, 7_I};
    constexpr let b = 5_I;
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    static_assert(std::is_same_v<decltype(c0), std::add_const_t<decltype(Crd{7_I, 12_I})>>);
    static_assert(std::is_same_v<decltype(c1), std::add_const_t<decltype(Crd{-3_I, 2_I})>>);
    static_assert(std::is_same_v<decltype(c2), std::add_const_t<decltype(Crd{10_I, 35_I})>>);
    static_assert(std::is_same_v<decltype(c3), std::add_const_t<decltype(Crd{7_I, 12_I})>>);
    static_assert(std::is_same_v<decltype(c4), std::add_const_t<decltype(Crd{3_I, -2_I})>>);
    static_assert(std::is_same_v<decltype(c5), std::add_const_t<decltype(Crd{10_I, 35_I})>>);
  }

  {
    constexpr let a = Crd{2_I, 7};
    constexpr let b = 5_I;
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    static_assert(std::is_same_v<decltype(c0), std::add_const_t<decltype(Crd{7_I, 12})>>);
    static_assert(std::is_same_v<decltype(c1), std::add_const_t<decltype(Crd{-3_I, 2})>>);
    static_assert(std::is_same_v<decltype(c2), std::add_const_t<decltype(Crd{10_I, 35})>>);
    static_assert(std::is_same_v<decltype(c3), std::add_const_t<decltype(Crd{7_I, 12})>>);
    static_assert(std::is_same_v<decltype(c4), std::add_const_t<decltype(Crd{3_I, -2})>>);
    static_assert(std::is_same_v<decltype(c5), std::add_const_t<decltype(Crd{10_I, 35})>>);
    EXPECT_EQ(get<1>(c0), 12);
    EXPECT_EQ(get<1>(c1), 2);
    EXPECT_EQ(get<1>(c2), 35);
    EXPECT_EQ(get<1>(c3), 12);
    EXPECT_EQ(get<1>(c4), -2);
    EXPECT_EQ(get<1>(c5), 35);
  }

  // 3D.
  {
    Crd3 a{2, 7, -5};
    Crd3 b{5, 11, -4};
    Crd3 c = a + b;
    Crd3 d = a - b;
    Crd3 e = a * b;
    expectCrd3(c, 7, 18, -9);
    expectCrd3(d, -3, -4, -1);
    expectCrd3(e, 10, 77, 20);
  }

  {
    constexpr let a = Crd{2_I, 7_I, -5_I};
    constexpr let b = Crd{5_I, 11_I, -4_I};
    constexpr let c = a + b;
    constexpr let d = a - b;
    constexpr let e = a * b;
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(Crd{7_I, 18_I, -9_I})>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(Crd{-3_I, -4_I, -1_I})>>);
    static_assert(std::is_same_v<decltype(e), std::add_const_t<decltype(Crd{10_I, 77_I, 20_I})>>);
  }

  {
    let a = Crd{2_I, 7, -5_I};
    let b = Crd{5_I, 11_I, -4_I};
    let c = a + b;
    let d = a - b;
    let e = a * b;
    static_assert(std::is_same_v<decltype(c), decltype(Crd{7_I, 18, -9_I})>);
    static_assert(std::is_same_v<decltype(d), decltype(Crd{-3_I, -4, -1_I})>);
    static_assert(std::is_same_v<decltype(e), decltype(Crd{10_I, 77, 20_I})>);
    EXPECT_EQ(get<1>(c), 18);
    EXPECT_EQ(get<1>(d), -4);
    EXPECT_EQ(get<1>(e), 77);
  }

  {
    Crd3 a{2, 7, -5};
    int b = 5;
    Crd3 c0 = a + b;
    Crd3 c1 = a - b;
    Crd3 c2 = a * b;
    Crd3 c3 = b + a;
    Crd3 c4 = b - a;
    Crd3 c5 = b * a;
    expectCrd3(c0, 7, 12, 0);
    expectCrd3(c1, -3, 2, -10);
    expectCrd3(c2, 10, 35, -25);
    expectCrd3(c3, 7, 12, 0);
    expectCrd3(c4, 3, -2, 10);
    expectCrd3(c5, 10, 35, -25);
  }

  {
    constexpr let a = Crd{2_I, 7_I, -5_I};
    constexpr let b = 5_I;
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    static_assert(std::is_same_v<decltype(c0), std::add_const_t<decltype(Crd{7_I, 12_I, 0_I})>>);
    static_assert(std::is_same_v<decltype(c1), std::add_const_t<decltype(Crd{-3_I, 2_I, -10_I})>>);
    static_assert(std::is_same_v<decltype(c2), std::add_const_t<decltype(Crd{10_I, 35_I, -25_I})>>);
    static_assert(std::is_same_v<decltype(c3), std::add_const_t<decltype(Crd{7_I, 12_I, 0_I})>>);
    static_assert(std::is_same_v<decltype(c4), std::add_const_t<decltype(Crd{3_I, -2_I, 10_I})>>);
    static_assert(std::is_same_v<decltype(c5), std::add_const_t<decltype(Crd{10_I, 35_I, -25_I})>>);
  }

  {
    constexpr let a = Crd{2_I, 7, -5_I};
    constexpr let b = 5_I;
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    static_assert(std::is_same_v<decltype(c0), std::add_const_t<decltype(Crd{7_I, 12, 0_I})>>);
    static_assert(std::is_same_v<decltype(c1), std::add_const_t<decltype(Crd{-3_I, 2, -10_I})>>);
    static_assert(std::is_same_v<decltype(c2), std::add_const_t<decltype(Crd{10_I, 35, -25_I})>>);
    static_assert(std::is_same_v<decltype(c3), std::add_const_t<decltype(Crd{7_I, 12, 0_I})>>);
    static_assert(std::is_same_v<decltype(c4), std::add_const_t<decltype(Crd{3_I, -2, 10_I})>>);
    static_assert(std::is_same_v<decltype(c5), std::add_const_t<decltype(Crd{10_I, 35, -25_I})>>);
    EXPECT_EQ(get<1>(c0), 12);
    EXPECT_EQ(get<1>(c1), 2);
    EXPECT_EQ(get<1>(c2), 35);
    EXPECT_EQ(get<1>(c3), 12);
    EXPECT_EQ(get<1>(c4), -2);
    EXPECT_EQ(get<1>(c5), 35);
  }
}

TEST(Layout, OperatorsFloat) {
  using Crd2 = Crd<float, float>;
  using Crd3 = Crd<float, float, float>;

  auto expectCrd2 = [](const Crd2 &crd, float x, float y) {
    EXPECT_FLOAT_EQ(get<0>(crd), x);
    EXPECT_FLOAT_EQ(get<1>(crd), y);
  };

  auto expectCrd3 = [](const Crd3 &crd, float x, float y, float z) {
    EXPECT_FLOAT_EQ(get<0>(crd), x);
    EXPECT_FLOAT_EQ(get<1>(crd), y);
    EXPECT_FLOAT_EQ(get<2>(crd), z);
  };

  {
    Crd2 a = cute::aria::layout::detail::FillCoords<float, float>(233.3F);
    Crd3 b = cute::aria::layout::detail::FillCoords<float, float, float>(233.3F);
    constexpr let c = cute::aria::layout::detail::FillCoords<C<233.3F>, C<233.3F>>(C<233.3F>{});
    constexpr let d = cute::aria::layout::detail::FillCoords<C<233.3F>, C<233.3F>, C<233.3F>>(C<233.3F>{});
    expectCrd2(a, 233.3F, 233.3F);
    expectCrd3(b, 233.3F, 233.3F, 233.3F);
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(Crd{C<233.3F>{}, C<233.3F>{}})>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(Crd{C<233.3F>{}, C<233.3F>{}, C<233.3F>{}})>>);
  }

  // 2D.
  {
    Crd2 a{2.1F, 7.2F};
    Crd2 b{5.3F, 11.4F};
    Crd2 c = a + b;
    Crd2 d = a - b;
    Crd2 e = a * b;
    expectCrd2(c, 7.4F, 18.6F);
    expectCrd2(d, -3.2F, -4.2F);
    expectCrd2(e, 11.13F, 82.08F);
  }

  {
    constexpr let a = Crd{C<2.1F>{}, C<7.2F>{}};
    constexpr let b = Crd{C<5.3F>{}, C<11.4F>{}};
    constexpr let c = a + b;
    constexpr let d = a - b;
    constexpr let e = a * b;
    expectCrd2(c, 7.4F, 18.6F);
    expectCrd2(d, -3.2F, -4.2F);
    expectCrd2(e, 11.13F, 82.08F);
  }

  {
    let a = Crd{C<2.1F>{}, 7.2F};
    let b = Crd{C<5.3F>{}, C<11.4F>{}};
    let c = a + b;
    let d = a - b;
    let e = a * b;
    expectCrd2(c, 7.4F, 18.6F);
    expectCrd2(d, -3.2F, -4.2F);
    expectCrd2(e, 11.13F, 82.08F);
  }

  {
    Crd2 a{2.1F, 7.2F};
    float b = 5.3F;
    Crd2 c0 = a + b;
    Crd2 c1 = a - b;
    Crd2 c2 = a * b;
    Crd2 c3 = b + a;
    Crd2 c4 = b - a;
    Crd2 c5 = b * a;
    expectCrd2(c0, 7.4F, 12.5F);
    expectCrd2(c1, -3.2F, 1.9F);
    expectCrd2(c2, 11.13F, 38.16F);
    expectCrd2(c3, 7.4F, 12.5F);
    expectCrd2(c4, 3.2F, -1.9F);
    expectCrd2(c5, 11.13F, 38.16F);
  }

  {
    constexpr let a = Crd{C<2.1F>{}, C<7.2F>{}};
    constexpr let b = C<5.3F>{};
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    expectCrd2(c0, 7.4F, 12.5F);
    expectCrd2(c1, -3.2F, 1.9F);
    expectCrd2(c2, 11.13F, 38.16F);
    expectCrd2(c3, 7.4F, 12.5F);
    expectCrd2(c4, 3.2F, -1.9F);
    expectCrd2(c5, 11.13F, 38.16F);
  }

  {
    constexpr let a = Crd{C<2.1F>{}, 7.2F};
    constexpr let b = C<5.3F>{};
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    expectCrd2(c0, 7.4F, 12.5F);
    expectCrd2(c1, -3.2F, 1.9F);
    expectCrd2(c2, 11.13F, 38.16F);
    expectCrd2(c3, 7.4F, 12.5F);
    expectCrd2(c4, 3.2F, -1.9F);
    expectCrd2(c5, 11.13F, 38.16F);
  }

  // 3D.
  {
    Crd3 a{2.1F, 7.2F, -5.5F};
    Crd3 b{5.3F, 11.4F, -4.5F};
    Crd3 c = a + b;
    Crd3 d = a - b;
    Crd3 e = a * b;
    expectCrd3(c, 7.4F, 18.6F, -10.0F);
    expectCrd3(d, -3.2F, -4.2F, -1.0F);
    expectCrd3(e, 11.13F, 82.08F, 24.75F);
  }

  {
    constexpr let a = Crd{C<2.1F>{}, C<7.2F>{}, C<-5.5F>{}};
    constexpr let b = Crd{C<5.3F>{}, C<11.4F>{}, C<-4.5F>{}};
    constexpr let c = a + b;
    constexpr let d = a - b;
    constexpr let e = a * b;
    expectCrd3(c, 7.4F, 18.6F, -10.0F);
    expectCrd3(d, -3.2F, -4.2F, -1.0F);
    expectCrd3(e, 11.13F, 82.08F, 24.75F);
  }

  {
    let a = Crd{C<2.1F>{}, 7.2F, C<-5.5F>{}};
    let b = Crd{C<5.3F>{}, C<11.4F>{}, C<-4.5F>{}};
    let c = a + b;
    let d = a - b;
    let e = a * b;
    expectCrd3(c, 7.4F, 18.6F, -10.0F);
    expectCrd3(d, -3.2F, -4.2F, -1.0F);
    expectCrd3(e, 11.13F, 82.08F, 24.75F);
  }

  {
    Crd3 a{2.1F, 7.2F, -5.5F};
    float b = 5.3F;
    Crd3 c0 = a + b;
    Crd3 c1 = a - b;
    Crd3 c2 = a * b;
    Crd3 c3 = b + a;
    Crd3 c4 = b - a;
    Crd3 c5 = b * a;
    expectCrd3(c0, 7.4F, 12.5F, -5.5F + 5.3F);
    expectCrd3(c1, -3.2F, 1.9F, -10.8F);
    expectCrd3(c2, 11.13F, 38.16F, -29.15F);
    expectCrd3(c3, 7.4F, 12.5F, -5.5F + 5.3F);
    expectCrd3(c4, 3.2F, -1.9F, 10.8F);
    expectCrd3(c5, 11.13F, 38.16F, -29.15F);
  }

  {
    constexpr let a = Crd{C<2.1F>{}, C<7.2F>{}, C<-5.5F>{}};
    constexpr let b = C<5.3F>{};
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    expectCrd3(c0, 7.4F, 12.5F, -5.5F + 5.3F);
    expectCrd3(c1, -3.2F, 1.9F, -10.8F);
    expectCrd3(c2, 11.13F, 38.16F, -29.15F);
    expectCrd3(c3, 7.4F, 12.5F, -5.5F + 5.3F);
    expectCrd3(c4, 3.2F, -1.9F, 10.8F);
    expectCrd3(c5, 11.13F, 38.16F, -29.15F);
  }

  {
    constexpr let a = Crd{C<2.1F>{}, 7.2F, C<-5.5F>{}};
    constexpr let b = C<5.3F>{};
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    expectCrd3(c0, 7.4F, 12.5F, -5.5F + 5.3F);
    expectCrd3(c1, -3.2F, 1.9F, -10.8F);
    expectCrd3(c2, 11.13F, 38.16F, -29.15F);
    expectCrd3(c3, 7.4F, 12.5F, -5.5F + 5.3F);
    expectCrd3(c4, 3.2F, -1.9F, 10.8F);
    expectCrd3(c5, 11.13F, 38.16F, -29.15F);
  }
}

} // namespace ARIA
