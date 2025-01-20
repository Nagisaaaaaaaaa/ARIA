#pragma once

#include "ARIA/ForEach.h"

#include <cute/layout.hpp>

#include <array>

// Define CTAD for CuTe types.
namespace cute {

template <typename... Ts>
tuple(Ts...) -> tuple<Ts...>;

}

//
//
//
//
//
namespace ARIA {

namespace layout::detail {

// Get the underlying arithmetic type.
// Examples:
//   int         -> int
//   const int   -> int
//   const int&  -> int
//   C<1>        -> int
//   const C<1>  -> int
//   const C<1>& -> int
template <typename T>
struct arithmetic_type;

template <typename T>
  requires(!ConstantArithmetic<std::decay_t<T>> && std::is_arithmetic_v<std::decay_t<T>>)
struct arithmetic_type<T> {
  using type = std::decay_t<T>;
};

template <typename T>
  requires(ConstantArithmetic<std::decay_t<T>>)
struct arithmetic_type<T> {
  using type = std::decay_t<decltype(std::decay_t<T>::value)>;
};

template <typename T>
using arithmetic_type_t = typename arithmetic_type<T>::type;

//
//
//
template <typename T, typename... Ts>
constexpr bool is_same_arithmetic_type_v = (std::is_same_v<arithmetic_type_t<T>, arithmetic_type_t<Ts>> && ...);

//
//
//
//
//
// Fetch implementations from CuTe.

using cute::rank;
using cute::rank_t;
using cute::rank_v;

using cute::get;

//
//
//
template <typename... Ts>
using Tup = cute::tuple<Ts...>;

template <typename... Ts>
  requires(!std::is_void_v<arithmetic_type_t<Ts>> && ...)
using Crd = cute::Coord<Ts...>;

template <typename... Ts>
ARIA_HOST_DEVICE constexpr Tup<Ts...> make_tup(const Ts &...ts) {
  return cute::make_tuple(ts...);
}

template <typename... Ts>
ARIA_HOST_DEVICE constexpr Crd<Ts...> make_crd(const Ts &...ts) {
  return cute::make_coord(ts...);
}

//
//
//
using cute::is_static;
using cute::is_static_v;

//
//
//
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

//! `is_layout_const_cosize` is implemented below.

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

//
//
//
//
//
// Cast `Crd` to `std::array`.
template <typename T, typename... Ts>
[[nodiscard]] ARIA_HOST_DEVICE static constexpr auto ToArray(const Crd<T, Ts...> &crd) {
  using value_type = arithmetic_type_t<T>;
  static_assert(is_same_arithmetic_type_v<T, Ts...>, "Element types of `Coord` should be \"as similar as possible\"");

  constexpr uint rank = rank_v<Crd<T, Ts...>>;

  std::array<value_type, rank> res;
  ForEach<rank>([&]<auto i>() { res[i] = get<i>(crd); });
  return res;
}

} // namespace layout::detail

} // namespace ARIA

//
//
//
//
//
// Arithmetic operators for coords.
//! WARNING: These operators should only be defined in `namespace cute` in order for
//! ADL to find the correct operators.

namespace cute {

namespace aria::layout::detail {

// Fill a `Coord` with a same value.
template <typename... Coords>
[[nodiscard]] ARIA_HOST_DEVICE constexpr Coord<Coords...>
FillCoords(const std::decay_t<decltype(get<0>(std::declval<Coord<Coords...>>()))> &v) {
  Coord<Coords...> c;
  ARIA::ForEach<sizeof...(Coords)>([&]<auto i>() {
    static_assert(std::is_same_v<std::decay_t<decltype(cute::get<i>(c))>, std::decay_t<decltype(v)>>,
                  "Element types of the `Coord` should be the same");
    cute::get<i>(c) = v;
  });

  return c;
}

} // namespace aria::layout::detail

template <typename... Coords0, typename... Coords1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator+(const Coord<Coords0...> &lhs, const Coord<Coords1...> &rhs) {
  //! `Coords0` is automatically required to have the same number of elements as `Coords1` at next line.
  Coord<decltype(std::declval<Coords0>() + std::declval<Coords1>())...> res;
  ARIA::ForEach<sizeof...(Coords0)>([&]<auto i>() { cute::get<i>(res) = cute::get<i>(lhs) + cute::get<i>(rhs); });
  return res;
}

template <typename... Coords0, typename... Coords1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator-(const Coord<Coords0...> &lhs, const Coord<Coords1...> &rhs) {
  Coord<decltype(std::declval<Coords0>() - std::declval<Coords1>())...> res;
  ARIA::ForEach<sizeof...(Coords0)>([&]<auto i>() { cute::get<i>(res) = cute::get<i>(lhs) - cute::get<i>(rhs); });
  return res;
}

template <typename... Coords0, typename... Coords1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator*(const Coord<Coords0...> &lhs, const Coord<Coords1...> &rhs) {
  Coord<decltype(std::declval<Coords0>() * std::declval<Coords1>())...> res;
  ARIA::ForEach<sizeof...(Coords0)>([&]<auto i>() { cute::get<i>(res) = cute::get<i>(lhs) * cute::get<i>(rhs); });
  return res;
}

template <typename... Coords0, typename Coords1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator+(const Coord<Coords0...> &lhs, const Coords1 &rhs) {
  //! `std::conditional_t<...>` at the following line is used to generate such a pack `<Coords1, Coords1, ...>`.
  return lhs + aria::layout::detail::FillCoords<std::conditional_t<true, Coords1, Coords0>...>(rhs);
}

template <typename... Coords0, typename Coords1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator+(const Coords1 &lhs, const Coord<Coords0...> &rhs) {
  return aria::layout::detail::FillCoords<std::conditional_t<true, Coords1, Coords0>...>(lhs) + rhs;
}

template <typename... Coords0, typename Coords1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator-(const Coord<Coords0...> &lhs, const Coords1 &rhs) {
  return lhs - aria::layout::detail::FillCoords<std::conditional_t<true, Coords1, Coords0>...>(rhs);
}

template <typename... Coords0, typename Coords1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator-(const Coords1 &lhs, const Coord<Coords0...> &rhs) {
  return aria::layout::detail::FillCoords<std::conditional_t<true, Coords1, Coords0>...>(lhs) - rhs;
}

template <typename... Coords0, typename Coords1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator*(const Coord<Coords0...> &lhs, const Coords1 &rhs) {
  return lhs * aria::layout::detail::FillCoords<std::conditional_t<true, Coords1, Coords0>...>(rhs);
}

template <typename... Coords0, typename Coords1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator*(const Coords1 &lhs, const Coord<Coords0...> &rhs) {
  return aria::layout::detail::FillCoords<std::conditional_t<true, Coords1, Coords0>...>(lhs) * rhs;
}

} // namespace cute
