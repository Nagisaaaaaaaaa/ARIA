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

TEST(Layout, Operators) {
  using Coord2 = Coord<int, int>;
  using Coord3 = Coord<int, int, int>;

  auto expectCoord2 = [](const Coord2 &coord, int x, int y) {
    EXPECT_EQ(cute::get<0>(coord), x);
    EXPECT_EQ(cute::get<1>(coord), y);
  };

  auto expectCoord3 = [](const Coord3 &coord, int x, int y, int z) {
    EXPECT_EQ(cute::get<0>(coord), x);
    EXPECT_EQ(cute::get<1>(coord), y);
    EXPECT_EQ(cute::get<2>(coord), z);
  };

  {
    Coord2 a = cute::aria::layout::detail::FillCoords<int, int>(233);
    Coord3 b = cute::aria::layout::detail::FillCoords<int, int, int>(233);
    constexpr let c = cute::aria::layout::detail::FillCoords<C<233>, C<233>>(C<233>{});
    constexpr let d = cute::aria::layout::detail::FillCoords<C<233>, C<233>, C<233>>(C<233>{});
    expectCoord2(a, 233, 233);
    expectCoord3(b, 233, 233, 233);
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(make_coord(233_I, 233_I))>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(make_coord(233_I, 233_I, 233_I))>>);
  }

  {
    Coord2 a{2, 7};
    Coord2 b{5, 11};
    Coord2 c = a + b;
    Coord2 d = a - b;
    Coord2 e = a * b;
    expectCoord2(c, 7, 18);
    expectCoord2(d, -3, -4);
    expectCoord2(e, 10, 77);
  }

  {
    constexpr let a = make_coord(2_I, 7_I);
    constexpr let b = make_coord(5_I, 11_I);
    constexpr let c = a + b;
    constexpr let d = a - b;
    constexpr let e = a * b;
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(make_coord(7_I, 18_I))>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(make_coord(-3_I, -4_I))>>);
    static_assert(std::is_same_v<decltype(e), std::add_const_t<decltype(make_coord(10_I, 77_I))>>);
  }

  {
    Coord2 a{2, 7};
    int b = 5;
    Coord2 c0 = a + b;
    Coord2 c1 = a - b;
    Coord2 c2 = a * b;
    Coord2 c3 = b + a;
    Coord2 c4 = b - a;
    Coord2 c5 = b * a;
    expectCoord2(c0, 7, 12);
    expectCoord2(c1, -3, 2);
    expectCoord2(c2, 10, 35);
    expectCoord2(c3, 7, 12);
    expectCoord2(c4, 3, -2);
    expectCoord2(c5, 10, 35);
  }

  {
    constexpr let a = make_coord(2_I, 7_I);
    constexpr let b = 5_I;
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    static_assert(std::is_same_v<decltype(c0), std::add_const_t<decltype(make_coord(7_I, 12_I))>>);
    static_assert(std::is_same_v<decltype(c1), std::add_const_t<decltype(make_coord(-3_I, 2_I))>>);
    static_assert(std::is_same_v<decltype(c2), std::add_const_t<decltype(make_coord(10_I, 35_I))>>);
    static_assert(std::is_same_v<decltype(c3), std::add_const_t<decltype(make_coord(7_I, 12_I))>>);
    static_assert(std::is_same_v<decltype(c4), std::add_const_t<decltype(make_coord(3_I, -2_I))>>);
    static_assert(std::is_same_v<decltype(c5), std::add_const_t<decltype(make_coord(10_I, 35_I))>>);
  }

  {
    Coord3 a{2, 7, -5};
    Coord3 b{5, 11, -4};
    Coord3 c = a + b;
    Coord3 d = a - b;
    Coord3 e = a * b;
    expectCoord3(c, 7, 18, -9);
    expectCoord3(d, -3, -4, -1);
    expectCoord3(e, 10, 77, 20);
  }

  {
    constexpr let a = make_coord(2_I, 7_I, -5_I);
    constexpr let b = make_coord(5_I, 11_I, -4_I);
    constexpr let c = a + b;
    constexpr let d = a - b;
    constexpr let e = a * b;
    static_assert(std::is_same_v<decltype(c), std::add_const_t<decltype(make_coord(7_I, 18_I, -9_I))>>);
    static_assert(std::is_same_v<decltype(d), std::add_const_t<decltype(make_coord(-3_I, -4_I, -1_I))>>);
    static_assert(std::is_same_v<decltype(e), std::add_const_t<decltype(make_coord(10_I, 77_I, 20_I))>>);
  }

  {
    Coord3 a{2, 7, -5};
    int b = 5;
    Coord3 c0 = a + b;
    Coord3 c1 = a - b;
    Coord3 c2 = a * b;
    Coord3 c3 = b + a;
    Coord3 c4 = b - a;
    Coord3 c5 = b * a;
    expectCoord3(c0, 7, 12, 0);
    expectCoord3(c1, -3, 2, -10);
    expectCoord3(c2, 10, 35, -25);
    expectCoord3(c3, 7, 12, 0);
    expectCoord3(c4, 3, -2, 10);
    expectCoord3(c5, 10, 35, -25);
  }

  {
    constexpr let a = make_coord(2_I, 7_I, -5_I);
    constexpr let b = 5_I;
    constexpr let c0 = a + b;
    constexpr let c1 = a - b;
    constexpr let c2 = a * b;
    constexpr let c3 = b + a;
    constexpr let c4 = b - a;
    constexpr let c5 = b * a;
    static_assert(std::is_same_v<decltype(c0), std::add_const_t<decltype(make_coord(7_I, 12_I, 0_I))>>);
    static_assert(std::is_same_v<decltype(c1), std::add_const_t<decltype(make_coord(-3_I, 2_I, -10_I))>>);
    static_assert(std::is_same_v<decltype(c2), std::add_const_t<decltype(make_coord(10_I, 35_I, -25_I))>>);
    static_assert(std::is_same_v<decltype(c3), std::add_const_t<decltype(make_coord(7_I, 12_I, 0_I))>>);
    static_assert(std::is_same_v<decltype(c4), std::add_const_t<decltype(make_coord(3_I, -2_I, 10_I))>>);
    static_assert(std::is_same_v<decltype(c5), std::add_const_t<decltype(make_coord(10_I, 35_I, -25_I))>>);
  }
}

} // namespace ARIA
