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

} // namespace ARIA
