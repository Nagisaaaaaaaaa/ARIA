#pragma once

/// \file
/// \brief A layout abstraction based on NVIDIA CuTe.
/// See https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md.
/// Please read the comments in `Tup.h` before continue.
///
/// \note We assume users have basic knowledge about CuTe.
/// If you are not familiar with CuTe, please read the tutorials of NVIDIA CuTe before continue:
/// 0. https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md.
/// 1. https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md.
/// 2. ...

//
//
//
//
//
#include "ARIA/detail/LayoutImpl.h"

namespace ARIA {

using layout::detail::Shape;

using layout::detail::Stride;

using layout::detail::make_shape;

using layout::detail::make_stride;

//
//
//
//
//
using layout::detail::Layout;

using layout::detail::make_layout;

using layout::detail::LayoutLeft; // Column major.

using layout::detail::LayoutRight; // Row major.

/// \brief Make a column major (left) or row major (right) layout.
/// `Shape` of the layout is defined by the input `coords`.
/// `Stride` of the layout is created automatically.
///
/// \tparam TMajor Optional.
/// Whether the layout is column major (left) or row major (right).
/// Should be either `LayoutLeft` or `LayoutRight`.
/// It is set to `LayoutLeft` by default.
///
/// \example ```cpp
/// auto layout0 = make_layout_major<LayoutLeft>(2, 4);
/// auto layout1 = make_layout_major(3, 5); // `LayoutLeft` by default.
/// ```
using layout::detail::make_layout_major;

/// \brief Whether the given type is a layout type.
using layout::detail::is_layout;

/// \brief Whether the given type is a layout type.
using layout::detail::is_layout_v;

//
//
//
//
//
using layout::detail::size;

using layout::detail::size_v;

/// \brief Safely get the cosize (maybe at compile time).
///
/// \warning `cosize` may throw runtime-errors when `size` equals to 0.
/// Always use this one instead.
///
/// Defining this function as an overload of `cosize` is unsafe, because
/// ADL may find the wrong one.
using layout::detail::cosize_safe;

/// \brief Safely get the cosize at compile time.
///
/// \warning `cosize_v` may fail to compile when `size_v` equals to 0.
/// Always use this one instead.
using layout::detail::cosize_safe_v;

//
//
//
//
//
/// \see is_layout_const_size_v
using layout::detail::is_layout_const_size;

/// \brief Whether `size` of `TLayout` is a compile-time value or not.
/// If yes, this means that `size<0>`, `size<1>`, and etc. are all compile-time values.
///
/// \example ```cpp
/// static_assert(is_layout_const_size_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
///                                                           make_stride(_4{}, make_stride(_2{}, _1{}))))>);
/// static_assert(!is_layout_const_size_v<decltype(make_layout(make_shape(2, make_shape(_2{}, _2{})),
///                                                            make_stride(_4{}, make_stride(_2{}, _1{}))))>);
/// ```
using layout::detail::is_layout_const_size_v;

/// \see is_layout_const_size_at_v
using layout::detail::is_layout_const_size_at;

/// \brief Whether `size<i>` of `TLayout` is a compile-time value or not.
///
/// \example ```cpp
/// static_assert(!is_layout_const_size_at_v<decltype(make_layout(make_shape(2, make_shape(_2{}, _2{})),
///                                                               make_stride(_4{}, make_stride(_2{}, _1{})))),
///                                          0>);
/// static_assert(is_layout_const_size_at_v<decltype(make_layout(make_shape(2, make_shape(_2{}, _2{})),
///                                                              make_stride(_4{}, make_stride(_2{}, _1{})))),
///                                         1>);
/// ```
using layout::detail::is_layout_const_size_at_v;

/// \see is_layout_const_cosize_safe_v
using layout::detail::is_layout_const_cosize_safe;

/// \brief Whether `cosize_safe` of `TLayout` is a compile-time value or not.
/// If yes, this means that the tensors defined by this layout can be stored in a `std::array`.
///
/// \example ```cpp
/// static_assert(is_layout_const_cosize_safe_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
///                                                                  make_stride(_4{}, make_stride(_2{}, _1{}))))>);
/// static_assert(!is_layout_const_cosize_safe_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
///                                                                   make_stride(4, make_stride(_2{}, _1{}))))>);
/// ```
using layout::detail::is_layout_const_cosize_safe_v;

//
//
//
//
//
/// \brief Co-layouts are the ones whose `size` and `cosize` are the same.
///
/// \example ```cpp
/// static_assert(is_co_layout(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
///                                        make_stride(_4{}, make_stride(_2{}, _1{})))));
/// EXPECT_TRUE(is_co_layout(make_layout(make_shape(2, make_shape(_2{}, _2{})),
///                                      make_stride(_4{}, make_stride(_2{}, _1{})))));
/// ```
using layout::detail::is_co_layout;

/// \brief Co-layouts are the ones whose `size` and `cosize` are the same.
///
/// \example ```cpp
/// static_assert(is_co_layout_v<decltype(make_layout(make_shape(_2{}, make_shape(_2{}, _2{})),
///                                                   make_stride(_4{}, make_stride(_2{}, _1{}))))>);
/// ```
using layout::detail::is_co_layout_v;

/// \brief Co-layouts are the ones whose `size` and `cosize` are the same.
///
/// \example ```cpp
/// template <CoLayout TCoLayout>
/// ...
/// ```
using layout::detail::CoLayout;

} // namespace ARIA
