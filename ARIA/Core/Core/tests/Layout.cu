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

  // Make tec.
  {
    Tec v{1, 2.0F, C<3U>{}, C<4.0>{}};
    EXPECT_EQ(get<0>(v), 1);
    EXPECT_EQ(get<1>(v), 2.0F);
    static_assert(get<2>(v) == C<3U>{});
    static_assert(get<3>(v) == C<4.0>{});
  }

  {
    constexpr Tec1 v1{1};
    constexpr Tec2 v2{1, 2.0F};
    constexpr Tec3 v3{1, 2.0F, C<3U>{}};
    constexpr Tec4 v4{1, 2.0F, C<3U>{}, C<4.0>{}};
  }

  {
    constexpr Teci vi{1, C<2>{}, 3, C<4>{}, 5};
    constexpr Tecu vu{1U, C<2U>{}, 3U, C<4U>{}, 5U};
    constexpr Tecf vf{1.0F, C<2.0F>{}, 3.0F, C<4.0F>{}, 5.0F};
    constexpr Tecd vd{1.0, C<2.0>{}, 3.0, C<4.0>{}, 5.0};
    constexpr Tecr vr{1_R, C<2_R>{}, 3_R, C<4_R>{}, 5_R};
  }

  {
    constexpr Tec1i vi_0{1};
    constexpr Tec1u vu_0{1U};
    constexpr Tec1f vf_0{1.0F};
    constexpr Tec1d vd_0{1.0};
    constexpr Tec1r vr_0{1_R};

    constexpr Tec1i vi_1{C<1>{}};
    constexpr Tec1u vu_1{C<1U>{}};
    constexpr Tec1f vf_1{C<1.0F>{}};
    constexpr Tec1d vd_1{C<1.0>{}};
    constexpr Tec1r vr_1{C<1_R>{}};
  }

  {
    constexpr Tec2i vi_00{1, 2};
    constexpr Tec2u vu_00{1U, 2U};
    constexpr Tec2f vf_00{1.0F, 2.0F};
    constexpr Tec2d vd_00{1.0, 2.0};
    constexpr Tec2r vr_00{1_R, 2_R};

    constexpr Tec2i vi_10{C<1>{}, 2};
    constexpr Tec2u vu_10{C<1U>{}, 2U};
    constexpr Tec2f vf_10{C<1.0F>{}, 2.0F};
    constexpr Tec2d vd_10{C<1.0>{}, 2.0};
    constexpr Tec2r vr_10{C<1_R>{}, 2_R};

    constexpr Tec2i vi_01{1, C<2>{}};
    constexpr Tec2u vu_01{1U, C<2U>{}};
    constexpr Tec2f vf_01{1.0F, C<2.0F>{}};
    constexpr Tec2d vd_01{1.0, C<2.0>{}};
    constexpr Tec2r vr_01{1_R, C<2_R>{}};

    constexpr Tec2i vi_11{C<1>{}, C<2>{}};
    constexpr Tec2u vu_11{C<1U>{}, C<2U>{}};
    constexpr Tec2f vf_11{C<1.0F>{}, C<2.0F>{}};
    constexpr Tec2d vd_11{C<1.0>{}, C<2.0>{}};
    constexpr Tec2r vr_11{C<1_R>{}, C<2_R>{}};
  }

  static_assert(rank(Tec{}) == 0);
  static_assert(rank(Tec{0}) == 1);
  static_assert(rank(Tec{_0{}}) == 1);
  static_assert(rank(Tec{0, 1}) == 2);
  static_assert(rank(Tec{_0{}, 1}) == 2);
  static_assert(rank(Tec{0, _1{}}) == 2);
  static_assert(rank(Tec{_0{}, _1{}}) == 2);

  static_assert(is_static_v<decltype(Tec{})>);
  static_assert(!is_static_v<decltype(Tec{0})>);
  static_assert(is_static_v<decltype(Tec{_0{}})>);
  static_assert(!is_static_v<decltype(Tec{0, 1})>);
  static_assert(!is_static_v<decltype(Tec{_0{}, 1})>);
  static_assert(!is_static_v<decltype(Tec{0, _1{}})>);
  static_assert(is_static_v<decltype(Tec{_0{}, _1{}})>);

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
    static_assert(ToArray(Tec{0}) == std::array{0});
    static_assert(ToArray(Tec{0_I}) == std::array{0});

    static_assert(ToArray(Tec{0U}) == std::array{0U});
    static_assert(ToArray(Tec{0_U}) == std::array{0U});

    static_assert(ToArray(Tec{0, 1}) == std::array{0, 1});
    static_assert(ToArray(Tec{0_I, 1}) == std::array{0, 1});
    static_assert(ToArray(Tec{0, 1_I}) == std::array{0, 1});
    static_assert(ToArray(Tec{0_I, 1_I}) == std::array{0, 1});

    static_assert(ToArray(Tec{0U, 1U}) == std::array{0U, 1U});
    static_assert(ToArray(Tec{0_U, 1U}) == std::array{0U, 1U});
    static_assert(ToArray(Tec{0U, 1_U}) == std::array{0U, 1U});
    static_assert(ToArray(Tec{0_U, 1_U}) == std::array{0U, 1U});

    static_assert(ToArray(Tec{0, 1, 2}) == std::array{0, 1, 2});
    static_assert(ToArray(Tec{0_I, 1, 2}) == std::array{0, 1, 2});
    static_assert(ToArray(Tec{0, 1_I, 2}) == std::array{0, 1, 2});
    static_assert(ToArray(Tec{0_I, 1_I, 2}) == std::array{0, 1, 2});
    static_assert(ToArray(Tec{0, 1, 2_I}) == std::array{0, 1, 2});
    static_assert(ToArray(Tec{0_I, 1, 2_I}) == std::array{0, 1, 2});
    static_assert(ToArray(Tec{0, 1_I, 2_I}) == std::array{0, 1, 2});
    static_assert(ToArray(Tec{0_I, 1_I, 2_I}) == std::array{0, 1, 2});

    static_assert(ToArray(Tec{0U, 1U, 2U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Tec{0_U, 1U, 2U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Tec{0U, 1_U, 2U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Tec{0_U, 1_U, 2U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Tec{0U, 1U, 2_U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Tec{0_U, 1U, 2_U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Tec{0U, 1_U, 2_U}) == std::array{0U, 1U, 2U});
    static_assert(ToArray(Tec{0_U, 1_U, 2_U}) == std::array{0U, 1U, 2U});
  }
}

TEST(Layout, OperatorsInt) {
  using Tec2 = Tec<int, int>;
  using Tec3 = Tec<int, int, int>;

  auto expectTec2 = [](const Tec2 &tec, int x, int y) {
    EXPECT_EQ(get<0>(tec), x);
    EXPECT_EQ(get<1>(tec), y);
  };

  auto expectTec3 = [](const Tec3 &tec, int x, int y, int z) {
    EXPECT_EQ(get<0>(tec), x);
    EXPECT_EQ(get<1>(tec), y);
    EXPECT_EQ(get<2>(tec), z);
  };

  {
    Tec2 a = cute::aria::layout::detail::FillCoords<int, int>(233);
    Tec3 b = cute::aria::layout::detail::FillCoords<int, int, int>(233);
    constexpr let c = cute::aria::layout::detail::FillCoords<C<233>, C<233>>(C<233>{});
    constexpr let d = cute::aria::layout::detail::FillCoords<C<233>, C<233>, C<233>>(C<233>{});
    expectTec2(a, 233, 233);
    expectTec3(b, 233, 233, 233);
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(Tec{233_I, 233_I})>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(Tec{233_I, 233_I, 233_I})>>);
  }

  // 2D.
  {
    Tec2 a{2, 7};
    Tec2 b{5, 11};
    Tec2 c = a + b;
    Tec2 d = a - b;
    Tec2 e = a * b;
    expectTec2(c, 7, 18);
    expectTec2(d, -3, -4);
    expectTec2(e, 10, 77);
  }

  {
    constexpr let a = Tec{2_I, 7_I};
    constexpr let b = Tec{5_I, 11_I};
    constexpr let c = a + b;
    constexpr let d = a - b;
    constexpr let e = a * b;
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(Tec{7_I, 18_I})>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(Tec{-3_I, -4_I})>>);
    static_assert(std::is_same_v<decltype(e), std::add_const_t<decltype(Tec{10_I, 77_I})>>);
  }

  {
    let a = Tec{2_I, 7};
    let b = Tec{5_I, 11_I};
    let c = a + b;
    let d = a - b;
    let e = a * b;
    static_assert(std::is_same_v<decltype(c), decltype(Tec{7_I, 18})>);
    static_assert(std::is_same_v<decltype(d), decltype(Tec{-3_I, -4})>);
    static_assert(std::is_same_v<decltype(e), decltype(Tec{10_I, 77})>);
    EXPECT_EQ(get<1>(c), 18);
    EXPECT_EQ(get<1>(d), -4);
    EXPECT_EQ(get<1>(e), 77);
  }

  {
    Tec2 a{2, 7};
    int b = 5;
    Tec2 c0 = a + b;
    Tec2 c1 = a - b;
    Tec2 c2 = a * b;
    Tec2 c3 = b + a;
    Tec2 c4 = b - a;
    Tec2 c5 = b * a;
    expectTec2(c0, 7, 12);
    expectTec2(c1, -3, 2);
    expectTec2(c2, 10, 35);
    expectTec2(c3, 7, 12);
    expectTec2(c4, 3, -2);
    expectTec2(c5, 10, 35);
  }

  {
    constexpr let a = Tec{2_I, 7_I};
    constexpr let b = 5_I;
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    static_assert(std::is_same_v<decltype(c0), std::add_const_t<decltype(Tec{7_I, 12_I})>>);
    static_assert(std::is_same_v<decltype(c1), std::add_const_t<decltype(Tec{-3_I, 2_I})>>);
    static_assert(std::is_same_v<decltype(c2), std::add_const_t<decltype(Tec{10_I, 35_I})>>);
    static_assert(std::is_same_v<decltype(c3), std::add_const_t<decltype(Tec{7_I, 12_I})>>);
    static_assert(std::is_same_v<decltype(c4), std::add_const_t<decltype(Tec{3_I, -2_I})>>);
    static_assert(std::is_same_v<decltype(c5), std::add_const_t<decltype(Tec{10_I, 35_I})>>);
  }

  {
    constexpr let a = Tec{2_I, 7};
    constexpr let b = 5_I;
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    static_assert(std::is_same_v<decltype(c0), std::add_const_t<decltype(Tec{7_I, 12})>>);
    static_assert(std::is_same_v<decltype(c1), std::add_const_t<decltype(Tec{-3_I, 2})>>);
    static_assert(std::is_same_v<decltype(c2), std::add_const_t<decltype(Tec{10_I, 35})>>);
    static_assert(std::is_same_v<decltype(c3), std::add_const_t<decltype(Tec{7_I, 12})>>);
    static_assert(std::is_same_v<decltype(c4), std::add_const_t<decltype(Tec{3_I, -2})>>);
    static_assert(std::is_same_v<decltype(c5), std::add_const_t<decltype(Tec{10_I, 35})>>);
    EXPECT_EQ(get<1>(c0), 12);
    EXPECT_EQ(get<1>(c1), 2);
    EXPECT_EQ(get<1>(c2), 35);
    EXPECT_EQ(get<1>(c3), 12);
    EXPECT_EQ(get<1>(c4), -2);
    EXPECT_EQ(get<1>(c5), 35);
  }

  // 3D.
  {
    Tec3 a{2, 7, -5};
    Tec3 b{5, 11, -4};
    Tec3 c = a + b;
    Tec3 d = a - b;
    Tec3 e = a * b;
    expectTec3(c, 7, 18, -9);
    expectTec3(d, -3, -4, -1);
    expectTec3(e, 10, 77, 20);
  }

  {
    constexpr let a = Tec{2_I, 7_I, -5_I};
    constexpr let b = Tec{5_I, 11_I, -4_I};
    constexpr let c = a + b;
    constexpr let d = a - b;
    constexpr let e = a * b;
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(Tec{7_I, 18_I, -9_I})>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(Tec{-3_I, -4_I, -1_I})>>);
    static_assert(std::is_same_v<decltype(e), std::add_const_t<decltype(Tec{10_I, 77_I, 20_I})>>);
  }

  {
    let a = Tec{2_I, 7, -5_I};
    let b = Tec{5_I, 11_I, -4_I};
    let c = a + b;
    let d = a - b;
    let e = a * b;
    static_assert(std::is_same_v<decltype(c), decltype(Tec{7_I, 18, -9_I})>);
    static_assert(std::is_same_v<decltype(d), decltype(Tec{-3_I, -4, -1_I})>);
    static_assert(std::is_same_v<decltype(e), decltype(Tec{10_I, 77, 20_I})>);
    EXPECT_EQ(get<1>(c), 18);
    EXPECT_EQ(get<1>(d), -4);
    EXPECT_EQ(get<1>(e), 77);
  }

  {
    Tec3 a{2, 7, -5};
    int b = 5;
    Tec3 c0 = a + b;
    Tec3 c1 = a - b;
    Tec3 c2 = a * b;
    Tec3 c3 = b + a;
    Tec3 c4 = b - a;
    Tec3 c5 = b * a;
    expectTec3(c0, 7, 12, 0);
    expectTec3(c1, -3, 2, -10);
    expectTec3(c2, 10, 35, -25);
    expectTec3(c3, 7, 12, 0);
    expectTec3(c4, 3, -2, 10);
    expectTec3(c5, 10, 35, -25);
  }

  {
    constexpr let a = Tec{2_I, 7_I, -5_I};
    constexpr let b = 5_I;
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    static_assert(std::is_same_v<decltype(c0), std::add_const_t<decltype(Tec{7_I, 12_I, 0_I})>>);
    static_assert(std::is_same_v<decltype(c1), std::add_const_t<decltype(Tec{-3_I, 2_I, -10_I})>>);
    static_assert(std::is_same_v<decltype(c2), std::add_const_t<decltype(Tec{10_I, 35_I, -25_I})>>);
    static_assert(std::is_same_v<decltype(c3), std::add_const_t<decltype(Tec{7_I, 12_I, 0_I})>>);
    static_assert(std::is_same_v<decltype(c4), std::add_const_t<decltype(Tec{3_I, -2_I, 10_I})>>);
    static_assert(std::is_same_v<decltype(c5), std::add_const_t<decltype(Tec{10_I, 35_I, -25_I})>>);
  }

  {
    constexpr let a = Tec{2_I, 7, -5_I};
    constexpr let b = 5_I;
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    static_assert(std::is_same_v<decltype(c0), std::add_const_t<decltype(Tec{7_I, 12, 0_I})>>);
    static_assert(std::is_same_v<decltype(c1), std::add_const_t<decltype(Tec{-3_I, 2, -10_I})>>);
    static_assert(std::is_same_v<decltype(c2), std::add_const_t<decltype(Tec{10_I, 35, -25_I})>>);
    static_assert(std::is_same_v<decltype(c3), std::add_const_t<decltype(Tec{7_I, 12, 0_I})>>);
    static_assert(std::is_same_v<decltype(c4), std::add_const_t<decltype(Tec{3_I, -2, 10_I})>>);
    static_assert(std::is_same_v<decltype(c5), std::add_const_t<decltype(Tec{10_I, 35, -25_I})>>);
    EXPECT_EQ(get<1>(c0), 12);
    EXPECT_EQ(get<1>(c1), 2);
    EXPECT_EQ(get<1>(c2), 35);
    EXPECT_EQ(get<1>(c3), 12);
    EXPECT_EQ(get<1>(c4), -2);
    EXPECT_EQ(get<1>(c5), 35);
  }
}

TEST(Layout, OperatorsFloat) {
  using Tec2 = Tec<float, float>;
  using Tec3 = Tec<float, float, float>;

  auto expectTec2 = [](const Tec2 &tec, float x, float y) {
    EXPECT_FLOAT_EQ(get<0>(tec), x);
    EXPECT_FLOAT_EQ(get<1>(tec), y);
  };

  auto expectTec3 = [](const Tec3 &tec, float x, float y, float z) {
    EXPECT_FLOAT_EQ(get<0>(tec), x);
    EXPECT_FLOAT_EQ(get<1>(tec), y);
    EXPECT_FLOAT_EQ(get<2>(tec), z);
  };

  {
    Tec2 a = cute::aria::layout::detail::FillCoords<float, float>(233.3F);
    Tec3 b = cute::aria::layout::detail::FillCoords<float, float, float>(233.3F);
    constexpr let c = cute::aria::layout::detail::FillCoords<C<233.3F>, C<233.3F>>(C<233.3F>{});
    constexpr let d = cute::aria::layout::detail::FillCoords<C<233.3F>, C<233.3F>, C<233.3F>>(C<233.3F>{});
    expectTec2(a, 233.3F, 233.3F);
    expectTec3(b, 233.3F, 233.3F, 233.3F);
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(Tec{C<233.3F>{}, C<233.3F>{}})>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(Tec{C<233.3F>{}, C<233.3F>{}, C<233.3F>{}})>>);
  }

  // 2D.
  {
    Tec2 a{2.1F, 7.2F};
    Tec2 b{5.3F, 11.4F};
    Tec2 c = a + b;
    Tec2 d = a - b;
    Tec2 e = a * b;
    expectTec2(c, 7.4F, 18.6F);
    expectTec2(d, -3.2F, -4.2F);
    expectTec2(e, 11.13F, 82.08F);
  }

  {
    constexpr let a = Tec{C<2.1F>{}, C<7.2F>{}};
    constexpr let b = Tec{C<5.3F>{}, C<11.4F>{}};
    constexpr let c = a + b;
    constexpr let d = a - b;
    constexpr let e = a * b;
    expectTec2(c, 7.4F, 18.6F);
    expectTec2(d, -3.2F, -4.2F);
    expectTec2(e, 11.13F, 82.08F);
  }

  {
    let a = Tec{C<2.1F>{}, 7.2F};
    let b = Tec{C<5.3F>{}, C<11.4F>{}};
    let c = a + b;
    let d = a - b;
    let e = a * b;
    expectTec2(c, 7.4F, 18.6F);
    expectTec2(d, -3.2F, -4.2F);
    expectTec2(e, 11.13F, 82.08F);
  }

  {
    Tec2 a{2.1F, 7.2F};
    float b = 5.3F;
    Tec2 c0 = a + b;
    Tec2 c1 = a - b;
    Tec2 c2 = a * b;
    Tec2 c3 = b + a;
    Tec2 c4 = b - a;
    Tec2 c5 = b * a;
    expectTec2(c0, 7.4F, 12.5F);
    expectTec2(c1, -3.2F, 1.9F);
    expectTec2(c2, 11.13F, 38.16F);
    expectTec2(c3, 7.4F, 12.5F);
    expectTec2(c4, 3.2F, -1.9F);
    expectTec2(c5, 11.13F, 38.16F);
  }

  {
    constexpr let a = Tec{C<2.1F>{}, C<7.2F>{}};
    constexpr let b = C<5.3F>{};
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    expectTec2(c0, 7.4F, 12.5F);
    expectTec2(c1, -3.2F, 1.9F);
    expectTec2(c2, 11.13F, 38.16F);
    expectTec2(c3, 7.4F, 12.5F);
    expectTec2(c4, 3.2F, -1.9F);
    expectTec2(c5, 11.13F, 38.16F);
  }

  {
    constexpr let a = Tec{C<2.1F>{}, 7.2F};
    constexpr let b = C<5.3F>{};
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    expectTec2(c0, 7.4F, 12.5F);
    expectTec2(c1, -3.2F, 1.9F);
    expectTec2(c2, 11.13F, 38.16F);
    expectTec2(c3, 7.4F, 12.5F);
    expectTec2(c4, 3.2F, -1.9F);
    expectTec2(c5, 11.13F, 38.16F);
  }

  // 3D.
  {
    Tec3 a{2.1F, 7.2F, -5.5F};
    Tec3 b{5.3F, 11.4F, -4.5F};
    Tec3 c = a + b;
    Tec3 d = a - b;
    Tec3 e = a * b;
    expectTec3(c, 7.4F, 18.6F, -10.0F);
    expectTec3(d, -3.2F, -4.2F, -1.0F);
    expectTec3(e, 11.13F, 82.08F, 24.75F);
  }

  {
    constexpr let a = Tec{C<2.1F>{}, C<7.2F>{}, C<-5.5F>{}};
    constexpr let b = Tec{C<5.3F>{}, C<11.4F>{}, C<-4.5F>{}};
    constexpr let c = a + b;
    constexpr let d = a - b;
    constexpr let e = a * b;
    expectTec3(c, 7.4F, 18.6F, -10.0F);
    expectTec3(d, -3.2F, -4.2F, -1.0F);
    expectTec3(e, 11.13F, 82.08F, 24.75F);
  }

  {
    let a = Tec{C<2.1F>{}, 7.2F, C<-5.5F>{}};
    let b = Tec{C<5.3F>{}, C<11.4F>{}, C<-4.5F>{}};
    let c = a + b;
    let d = a - b;
    let e = a * b;
    expectTec3(c, 7.4F, 18.6F, -10.0F);
    expectTec3(d, -3.2F, -4.2F, -1.0F);
    expectTec3(e, 11.13F, 82.08F, 24.75F);
  }

  {
    Tec3 a{2.1F, 7.2F, -5.5F};
    float b = 5.3F;
    Tec3 c0 = a + b;
    Tec3 c1 = a - b;
    Tec3 c2 = a * b;
    Tec3 c3 = b + a;
    Tec3 c4 = b - a;
    Tec3 c5 = b * a;
    expectTec3(c0, 7.4F, 12.5F, -5.5F + 5.3F);
    expectTec3(c1, -3.2F, 1.9F, -10.8F);
    expectTec3(c2, 11.13F, 38.16F, -29.15F);
    expectTec3(c3, 7.4F, 12.5F, -5.5F + 5.3F);
    expectTec3(c4, 3.2F, -1.9F, 10.8F);
    expectTec3(c5, 11.13F, 38.16F, -29.15F);
  }

  {
    constexpr let a = Tec{C<2.1F>{}, C<7.2F>{}, C<-5.5F>{}};
    constexpr let b = C<5.3F>{};
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    expectTec3(c0, 7.4F, 12.5F, -5.5F + 5.3F);
    expectTec3(c1, -3.2F, 1.9F, -10.8F);
    expectTec3(c2, 11.13F, 38.16F, -29.15F);
    expectTec3(c3, 7.4F, 12.5F, -5.5F + 5.3F);
    expectTec3(c4, 3.2F, -1.9F, 10.8F);
    expectTec3(c5, 11.13F, 38.16F, -29.15F);
  }

  {
    constexpr let a = Tec{C<2.1F>{}, 7.2F, C<-5.5F>{}};
    constexpr let b = C<5.3F>{};
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    expectTec3(c0, 7.4F, 12.5F, -5.5F + 5.3F);
    expectTec3(c1, -3.2F, 1.9F, -10.8F);
    expectTec3(c2, 11.13F, 38.16F, -29.15F);
    expectTec3(c3, 7.4F, 12.5F, -5.5F + 5.3F);
    expectTec3(c4, 3.2F, -1.9F, 10.8F);
    expectTec3(c5, 11.13F, 38.16F, -29.15F);
  }
}

} // namespace ARIA
