#pragma once

#include "ARIA/ForEach.h"

#include <cute/int_tuple.hpp>

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
//
//
//
//
namespace ARIA {

namespace tup::detail {

// Get the underlying arithmetic domain.
// Examples:
//   int         -> int
//   const int   -> int
//   const int&  -> int
//   C<1>        -> int
//   const C<1>  -> int
//   const C<1>& -> int
//   std::string -> void
template <typename T>
struct arithmetic_domain {
  using type = void;
};

template <typename T>
  requires(!ConstantArithmetic<std::decay_t<T>> && std::is_arithmetic_v<std::decay_t<T>>)
struct arithmetic_domain<T> {
  using type = std::decay_t<T>;
};

template <typename T>
  requires(ConstantArithmetic<std::decay_t<T>>)
struct arithmetic_domain<T> {
  using type = std::decay_t<decltype(std::decay_t<T>::value)>;
};

template <typename T>
using arithmetic_domain_t = typename arithmetic_domain<T>::type;

template <typename T>
constexpr bool has_arithmetic_domain_v = !std::is_void_v<arithmetic_domain_t<T>>;

template <typename T, typename... Ts>
constexpr bool is_same_arithmetic_domain_v =
    has_arithmetic_domain_v<T> && (std::is_same_v<arithmetic_domain_t<T>, arithmetic_domain_t<Ts>> && ...);

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
  requires(has_arithmetic_domain_v<Ts> && ...)
using Tec = cute::tuple<Ts...>;

//
//
//
//
//
// TODO: Great efforts are made to bypass the MSVC bug.
template <typename... Ts>
  requires((sizeof...(Ts) == 1 && has_arithmetic_domain_v<Ts>) && ...)
using Tec1 = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 2 && has_arithmetic_domain_v<Ts>) && ...)
using Tec2 = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 3 && has_arithmetic_domain_v<Ts>) && ...)
using Tec3 = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 4 && has_arithmetic_domain_v<Ts>) && ...)
using Tec4 = cute::tuple<Ts...>;

//
//
//
template <typename... Ts>
  requires(std::is_same_v<arithmetic_domain_t<Ts>, int> && ...)
using Teci = cute::tuple<Ts...>;

template <typename... Ts>
  requires(std::is_same_v<arithmetic_domain_t<Ts>, uint> && ...)
using Tecu = cute::tuple<Ts...>;

template <typename... Ts>
  requires(std::is_same_v<arithmetic_domain_t<Ts>, float> && ...)
using Tecf = cute::tuple<Ts...>;

template <typename... Ts>
  requires(std::is_same_v<arithmetic_domain_t<Ts>, double> && ...)
using Tecd = cute::tuple<Ts...>;

template <typename... Ts>
  requires(std::is_same_v<arithmetic_domain_t<Ts>, Real> && ...)
using Tecr = cute::tuple<Ts...>;

//
//
//
template <typename... Ts>
  requires((sizeof...(Ts) == 1 && std::is_same_v<arithmetic_domain_t<Ts>, int>) && ...)
using Tec1i = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 1 && std::is_same_v<arithmetic_domain_t<Ts>, uint>) && ...)
using Tec1u = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 1 && std::is_same_v<arithmetic_domain_t<Ts>, float>) && ...)
using Tec1f = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 1 && std::is_same_v<arithmetic_domain_t<Ts>, double>) && ...)
using Tec1d = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 1 && std::is_same_v<arithmetic_domain_t<Ts>, Real>) && ...)
using Tec1r = cute::tuple<Ts...>;

//
//
//
template <typename... Ts>
  requires((sizeof...(Ts) == 2 && std::is_same_v<arithmetic_domain_t<Ts>, int>) && ...)
using Tec2i = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 2 && std::is_same_v<arithmetic_domain_t<Ts>, uint>) && ...)
using Tec2u = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 2 && std::is_same_v<arithmetic_domain_t<Ts>, float>) && ...)
using Tec2f = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 2 && std::is_same_v<arithmetic_domain_t<Ts>, double>) && ...)
using Tec2d = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 2 && std::is_same_v<arithmetic_domain_t<Ts>, Real>) && ...)
using Tec2r = cute::tuple<Ts...>;

//
//
//
template <typename... Ts>
  requires((sizeof...(Ts) == 3 && std::is_same_v<arithmetic_domain_t<Ts>, int>) && ...)
using Tec3i = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 3 && std::is_same_v<arithmetic_domain_t<Ts>, uint>) && ...)
using Tec3u = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 3 && std::is_same_v<arithmetic_domain_t<Ts>, float>) && ...)
using Tec3f = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 3 && std::is_same_v<arithmetic_domain_t<Ts>, double>) && ...)
using Tec3d = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 3 && std::is_same_v<arithmetic_domain_t<Ts>, Real>) && ...)
using Tec3r = cute::tuple<Ts...>;

//
//
//
template <typename... Ts>
  requires((sizeof...(Ts) == 4 && std::is_same_v<arithmetic_domain_t<Ts>, int>) && ...)
using Tec4i = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 4 && std::is_same_v<arithmetic_domain_t<Ts>, uint>) && ...)
using Tec4u = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 4 && std::is_same_v<arithmetic_domain_t<Ts>, float>) && ...)
using Tec4f = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 4 && std::is_same_v<arithmetic_domain_t<Ts>, double>) && ...)
using Tec4d = cute::tuple<Ts...>;

template <typename... Ts>
  requires((sizeof...(Ts) == 4 && std::is_same_v<arithmetic_domain_t<Ts>, Real>) && ...)
using Tec4r = cute::tuple<Ts...>;

//
//
//
// Whether the given type is `Tec<...>`.
template <typename T>
struct is_tec : std::false_type {};

template <typename... Ts>
struct is_tec<Tec<Ts...>> : std::true_type {};

template <typename T>
constexpr bool is_tec_v = is_tec<T>::value;

//
//
//
// Whether the given `Tec` has rank `r`.
template <typename TTec, auto r>
struct is_tec_r : std::false_type {};

template <typename... Ts, auto r>
  requires(sizeof...(Ts) == r)
struct is_tec_r<Tec<Ts...>, r> : std::true_type {};

template <typename TTec, auto r>
constexpr bool is_tec_r_v = is_tec_r<TTec, r>::value;

//
//
//
// Whether the given `Tec` has "arithmetic" type `T`.
template <typename TTec, typename T>
struct is_tec_t : std::false_type {};

template <typename... Ts, typename T>
  requires(std::is_same_v<arithmetic_domain_t<Ts>, T> && ...)
struct is_tec_t<Tec<Ts...>, T> : std::true_type {};

template <typename TTec, typename T>
constexpr bool is_tec_t_v = is_tec_t<TTec, T>::value;

//
//
//
// Whether the given `Tec` has "arithmetic" type `T` and rank `r`.
template <typename TTec, typename T, auto r>
struct is_tec_tr : std::false_type {};

template <typename... Ts, typename T, auto r>
  requires(sizeof...(Ts) == r && (std::is_same_v<arithmetic_domain_t<Ts>, T> && ...))
struct is_tec_tr<Tec<Ts...>, T, r> : std::true_type {};

template <typename TTec, typename T, auto r>
constexpr bool is_tec_tr_v = is_tec_tr<TTec, T, r>::value;

//
//
//
//
//
using cute::is_static;
using cute::is_static_v;

//
//
//
//
//
// Cast `Tec` to `std::array`.
template <typename T, typename... Ts>
[[nodiscard]] ARIA_HOST_DEVICE static constexpr auto ToArray(const Tec<T, Ts...> &tec) {
  static_assert(is_same_arithmetic_domain_v<T, Ts...>, "Element types of `Tec` should be \"as similar as possible\"");
  using value_type = arithmetic_domain_t<T>;

  constexpr uint rank = rank_v<Tec<T, Ts...>>;

  std::array<value_type, rank> res;
  ForEach<rank>([&]<auto i>() { res[i] = get<i>(tec); });
  return res;
}

} // namespace tup::detail

} // namespace ARIA

//
//
//
//
//
//
//
//
//
// Arithmetic operators for coords.
//! WARNING: These operators should only be defined in `namespace cute` in order for
//! ADL to find the correct operators.

namespace cute {

namespace aria::tup::detail {

// Fill a `Tec` with a same value.
template <typename... Ts>
[[nodiscard]] ARIA_HOST_DEVICE constexpr ARIA::tup::detail::Tec<Ts...>
FillTec(const std::decay_t<decltype(get<0>(std::declval<ARIA::tup::detail::Tec<Ts...>>()))> &v) {
  ARIA::tup::detail::Tec<Ts...> c;
  ARIA::ForEach<sizeof...(Ts)>([&]<auto i>() {
    static_assert(std::is_same_v<std::decay_t<decltype(cute::get<i>(c))>, std::decay_t<decltype(v)>>,
                  "Element types of the `Tec` should be the same");
    cute::get<i>(c) = v;
  });

  return c;
}

} // namespace aria::tup::detail

//
//
//
template <typename... Ts0, typename... Ts1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator+(const ARIA::tup::detail::Tec<Ts0...> &lhs,
                                                        const ARIA::tup::detail::Tec<Ts1...> &rhs) {
  //! `Ts0` is automatically required to have the same number of elements as `Ts1` at next line.
  ARIA::tup::detail::Tec<decltype(std::declval<Ts0>() + std::declval<Ts1>())...> res;
  ARIA::ForEach<sizeof...(Ts0)>([&]<auto i>() { cute::get<i>(res) = cute::get<i>(lhs) + cute::get<i>(rhs); });
  return res;
}

template <typename... Ts0, typename... Ts1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator-(const ARIA::tup::detail::Tec<Ts0...> &lhs,
                                                        const ARIA::tup::detail::Tec<Ts1...> &rhs) {
  ARIA::tup::detail::Tec<decltype(std::declval<Ts0>() - std::declval<Ts1>())...> res;
  ARIA::ForEach<sizeof...(Ts0)>([&]<auto i>() { cute::get<i>(res) = cute::get<i>(lhs) - cute::get<i>(rhs); });
  return res;
}

template <typename... Ts0, typename... Ts1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator*(const ARIA::tup::detail::Tec<Ts0...> &lhs,
                                                        const ARIA::tup::detail::Tec<Ts1...> &rhs) {
  ARIA::tup::detail::Tec<decltype(std::declval<Ts0>() * std::declval<Ts1>())...> res;
  ARIA::ForEach<sizeof...(Ts0)>([&]<auto i>() { cute::get<i>(res) = cute::get<i>(lhs) * cute::get<i>(rhs); });
  return res;
}

template <typename... Ts0, typename Ts1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator+(const ARIA::tup::detail::Tec<Ts0...> &lhs, const Ts1 &rhs) {
  //! `std::conditional_t<...>` at the following line is used to generate such a pack `<Ts1, Ts1, ...>`.
  return lhs + aria::tup::detail::FillTec<std::conditional_t<true, Ts1, Ts0>...>(rhs);
}

template <typename... Ts0, typename Ts1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator+(const Ts1 &lhs, const ARIA::tup::detail::Tec<Ts0...> &rhs) {
  return aria::tup::detail::FillTec<std::conditional_t<true, Ts1, Ts0>...>(lhs) + rhs;
}

template <typename... Ts0, typename Ts1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator-(const ARIA::tup::detail::Tec<Ts0...> &lhs, const Ts1 &rhs) {
  return lhs - aria::tup::detail::FillTec<std::conditional_t<true, Ts1, Ts0>...>(rhs);
}

template <typename... Ts0, typename Ts1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator-(const Ts1 &lhs, const ARIA::tup::detail::Tec<Ts0...> &rhs) {
  return aria::tup::detail::FillTec<std::conditional_t<true, Ts1, Ts0>...>(lhs) - rhs;
}

template <typename... Ts0, typename Ts1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator*(const ARIA::tup::detail::Tec<Ts0...> &lhs, const Ts1 &rhs) {
  return lhs * aria::tup::detail::FillTec<std::conditional_t<true, Ts1, Ts0>...>(rhs);
}

template <typename... Ts0, typename Ts1>
[[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator*(const Ts1 &lhs, const ARIA::tup::detail::Tec<Ts0...> &rhs) {
  return aria::tup::detail::FillTec<std::conditional_t<true, Ts1, Ts0>...>(lhs) * rhs;
}

} // namespace cute
