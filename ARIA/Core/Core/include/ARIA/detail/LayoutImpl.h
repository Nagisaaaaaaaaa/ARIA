#pragma once

#include "ARIA/Tup.h"

#include <cute/layout.hpp>

namespace ARIA {

namespace layout::detail {

using cute::Shape;
using cute::Stride;

using cute::make_shape;
using cute::make_stride;

//
//
//
//
//
using cute::Layout;

using cute::make_layout;

using cute::LayoutLeft;  // Column major.
using cute::LayoutRight; // Row major.

template <typename TMajor, typename... Coords>
ARIA_HOST_DEVICE decltype(auto) make_layout_major(Coords &&...coords) {
  return make_layout(make_shape(std::forward<Coords>(coords)...), TMajor{});
}

// CuTe uses `LayoutLeft` by default.
template <typename... Coords>
ARIA_HOST_DEVICE decltype(auto) make_layout_major(Coords &&...coords) {
  return make_layout(make_shape(std::forward<Coords>(coords)...));
}

// Whether the given type is a layout type.
using cute::is_layout;

// Whether the given type is a layout type.
template <typename T>
constexpr bool is_layout_v = is_layout<T>::value;

// Whether the given type is a layout type.
template <typename T>
concept LayoutType = is_layout_v<T>;

//
//
//
//
//
using cute::size;
using cute::size_v;

// Unsafely get the cosize.
template <LayoutType TLayout>
ARIA_HOST_DEVICE constexpr auto cosize_unsafe(const TLayout &layout) {
  return cute::cosize(layout);
}

template <LayoutType TLayout>
static constexpr auto cosize_unsafe_v = cute::cosize_v<TLayout>;

//! `cosize_safe` and `cosize_safe_v` are implemented below.

//
//
//
//
//
// Whether `size` of `TLayout` is a compile-time value or not.
template <LayoutType TLayout>
struct is_layout_const_size {
  static constexpr bool value = false;
};

template <LayoutType TLayout>
  requires ConstantIntegral<decltype(size(std::declval<TLayout>()))>
struct is_layout_const_size<TLayout> {
  static constexpr bool value = true;
};

template <LayoutType TLayout>
constexpr bool is_layout_const_size_v = is_layout_const_size<TLayout>::value;

//
//
//
// Whether `size<i>` of `TLayout` is a compile-time value or not.
template <LayoutType TLayout, uint i>
struct is_layout_const_size_at {
  static constexpr bool value = false;
};

template <LayoutType TLayout, uint i>
  requires ConstantIntegral<decltype(size<i>(std::declval<TLayout>()))>
struct is_layout_const_size_at<TLayout, i> {
  static constexpr bool value = true;
};

template <LayoutType TLayout, uint i>
constexpr bool is_layout_const_size_at_v = is_layout_const_size_at<TLayout, i>::value;

//
//
//
// Whether `cosize_unsafe` of `TLayout` is a compile-time value or not.
template <LayoutType TLayout>
struct is_layout_const_cosize_unsafe {
  static constexpr bool value = false;
};

template <LayoutType TLayout>
  requires ConstantIntegral<decltype(cosize_unsafe(std::declval<TLayout>()))>
struct is_layout_const_cosize_unsafe<TLayout> {
  static constexpr bool value = true;
};

template <LayoutType TLayout>
constexpr bool is_layout_const_cosize_unsafe_v = is_layout_const_cosize_unsafe<TLayout>::value;

//! `is_layout_const_cosize_safe` is implemented below.

//
//
//
//
//
// Safely get the cosize.
template <LayoutType TLayout>
struct cosize_safe_impl {};

//! `cosize_unsafe_v` may fail to compile when `size_v` equals to 0.
//! For these cases, set `cosize_safe_v` to 0.
template <LayoutType TLayout>
  requires(is_layout_const_size_v<TLayout> && size_v<TLayout> == 0)
struct cosize_safe_impl<TLayout> {
  static constexpr int value = 0;
};

//! The compile-time cosize can be safely fetched from `cosize_unsafe_v` only when:
//! 1. `size_v` exists.
//! 2. `size_v > 0`.
//! 3. `cosize_unsafe_v` exists.
template <LayoutType TLayout>
  requires(is_layout_const_size_v<TLayout> && size_v<TLayout> > 0 && is_layout_const_cosize_unsafe_v<TLayout>)
struct cosize_safe_impl<TLayout> {
  static constexpr int value = cosize_unsafe_v<TLayout>;
};

template <LayoutType TLayout>
static constexpr int cosize_safe_v = cosize_safe_impl<TLayout>::value;

//
//
//
template <LayoutType TLayout>
ARIA_HOST_DEVICE constexpr auto cosize_safe(const TLayout &layout) {
  if constexpr (is_layout_const_size_v<TLayout>) { // Const size.
    if constexpr (size_v<TLayout> == 0)            // Const and zero size.
      return 0;
    else // Const and non-zero size.
      return cosize_unsafe(layout);
  } else {                 // Non-const size.
    if (size(layout) == 0) // Non-const and zero size.
      return 0;
    else // Non-const and non-zero size.
      return cosize_unsafe(layout);
  }
}

//
//
//
//
//
// Whether `cosize_safe` of `TLayout` is a compile-time value or not.
template <LayoutType TLayout>
struct is_layout_const_cosize_safe {
  static constexpr bool value = false;
};

template <LayoutType TLayout>
  requires ConstantIntegral<decltype(cosize_safe(std::declval<TLayout>()))>
struct is_layout_const_cosize_safe<TLayout> {
  static constexpr bool value = true;
};

template <LayoutType TLayout>
constexpr bool is_layout_const_cosize_safe_v = is_layout_const_cosize_safe<TLayout>::value;

//
//
//
//
//
// Co-layouts are the ones whose `size` and `cosize` are the same.
template <LayoutType TLayout>
constexpr bool is_co_layout(const TLayout &layout) {
  return size(layout) == cosize_safe(layout);
}

template <LayoutType TLayout>
  requires is_layout_const_cosize_safe_v<TLayout>
constexpr bool is_co_layout_v = size_v<TLayout> == cosize_safe_v<TLayout>;

template <typename TLayout>
concept CoLayout = is_co_layout_v<TLayout>;

} // namespace layout::detail

} // namespace ARIA
