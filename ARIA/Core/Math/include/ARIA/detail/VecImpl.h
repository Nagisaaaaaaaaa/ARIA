#pragma once

#include "ARIA/Layout.h"
#include "ARIA/detail/MatImpl.h"

namespace ARIA {

namespace vec::detail {

// Similar to `Mat`.
template <typename T, auto size>
using Vec = Eigen::Vector<T, size>;

//
//
//
// Whether the given type is `Vec<T, ...>`.
template <typename T>
struct is_vec : std::false_type {};

template <typename T, auto size>
struct is_vec<Vec<T, size>> : std::true_type {};

template <typename T>
static constexpr bool is_vec_v = is_vec<T>::value;

//
//
//
// Whether the given type is `Vec<T, s>`.
template <typename T, auto s>
struct is_vec_s : std::false_type {};

template <typename T, auto s>
struct is_vec_s<Vec<T, s>, s> : std::true_type {};

template <typename T, auto s>
static constexpr bool is_vec_s_v = is_vec_s<T, s>::value;

//
//
//
//
//
//! `Eigen::Vector` is just a special case of `Eigen::Matrix`, so
//! they share the same member functions.
#define __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_VEC __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_MAT

#define __ARIA_PROP_PREFAB_VEC(accessGet, accessSet, specifiers, type, /*propName,*/...)                               \
  static_assert(ARIA::vec::detail::is_vec_v<std::decay_t<type>>,                                                       \
                "Type of the property should be `class Vec` in order to use this prefab");                             \
                                                                                                                       \
  ARIA_PROP_BEGIN(accessGet, accessSet, specifiers, type, /*propName,*/ __VA_ARGS__);                                  \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_VEC(specifiers, type);                                                       \
  ARIA_PROP_END

#define __ARIA_SUB_PROP_PREFAB_VEC(specifiers, type, /*propName,*/...)                                                 \
  static_assert(ARIA::vec::detail::is_vec_v<std::decay_t<type>>,                                                       \
                "Type of the property should be `class Vec` in order to use this prefab");                             \
                                                                                                                       \
  ARIA_SUB_PROP_BEGIN(specifiers, type, /*propName,*/ __VA_ARGS__);                                                    \
  __ARIA_PROP_AND_SUB_PROP_PREFAB_MEMBERS_VEC(specifiers, type);                                                       \
  ARIA_PROP_END

//
//
//
//
//
// Cast `Vec` to `Coord`.
template <typename T, auto n, auto i, typename... TValues>
ARIA_HOST_DEVICE static constexpr auto ToCoordImpl(const Vec<T, n> &vec, TValues &&...values) {
  if constexpr (i == 0)
    return make_coord(std::forward<TValues>(values)...);
  else
    return ToCoordImpl<T, n, i - 1>(vec, vec[i - 1], std::forward<TValues>(values)...);
}

template <typename T, auto n>
ARIA_HOST_DEVICE static constexpr auto ToCoord(const Vec<T, n> &vec) {
  return ToCoordImpl<T, n, n>(vec);
}

//
//
//
// Cast `Coord` to `Vec`.
template <typename T, typename... Ts>
ARIA_HOST_DEVICE static constexpr auto ToVec(const Coord<T, Ts...> &coord) {
  static_assert((std::is_same_v<T, Ts> && ...), "Element types of `Coord` should be the same");
  constexpr auto n = sizeof...(Ts) + 1;

  Vec<T, n> vec;
  ForEach<n>([&]<auto i>() { vec[i] = get<i>(coord); });
  return vec;
}

} // namespace vec::detail

} // namespace ARIA
