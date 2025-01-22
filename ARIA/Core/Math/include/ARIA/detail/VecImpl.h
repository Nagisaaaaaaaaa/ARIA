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
// Cast `Vec` to `Crd`.
template <typename T, auto n, auto i, typename... TValues>
[[nodiscard]] ARIA_HOST_DEVICE static constexpr auto ToCrdImpl(const Vec<T, n> &vec, TValues &&...values) {
  if constexpr (i == 0)
    return make_crd(std::forward<TValues>(values)...);
  else
    return ToCrdImpl<T, n, i - 1>(vec, vec[i - 1], std::forward<TValues>(values)...);
}

template <typename T, auto n>
[[nodiscard]] ARIA_HOST_DEVICE static constexpr auto ToCrd(const Vec<T, n> &vec) {
  return ToCrdImpl<T, n, n>(vec);
}

//
//
//
// Cast `Crd` to `Vec`.
template <typename T, typename... Ts>
[[nodiscard]] ARIA_HOST_DEVICE static constexpr auto ToVec(const Crd<T, Ts...> &crd) {
  static_assert(layout::detail::is_same_arithmetic_domain_v<T, Ts...>,
                "Element types of `Crd` should be \"as similar as possible\"");
  using value_type = layout::detail::arithmetic_domain_t<T>;

  constexpr uint rank = rank_v<Crd<T, Ts...>>;

  Vec<value_type, rank> res;
  ForEach<rank>([&]<auto i>() { res[i] = get<i>(crd); });
  return res;
}

} // namespace vec::detail

} // namespace ARIA
