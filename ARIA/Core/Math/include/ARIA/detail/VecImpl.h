#pragma once

#include "ARIA/Tup.h"
#include "ARIA/detail/MatImpl.h"

namespace ARIA {

namespace vec::detail {

// Similar to `Mat`.
template <typename T, auto size>
using Vec = Eigen::Vector<T, size>;

//
//
//
// Whether the given type is `Vec<...>`.
template <typename T>
struct is_vec : std::false_type {};

template <typename T, auto size>
struct is_vec<Vec<T, size>> : std::true_type {};

template <typename T>
static constexpr bool is_vec_v = is_vec<T>::value;

//
//
//
// Whether the given type is `Vec<..., s>`.
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
// Cast `Vec` to `Tec`.
template <typename T, auto n, auto i, typename... TValues>
[[nodiscard]] ARIA_HOST_DEVICE static constexpr auto ToTecImpl(const Vec<T, n> &vec, TValues &&...values) {
  if constexpr (i == 0)
    return Tec{std::forward<TValues>(values)...};
  else
    return ToTecImpl<T, n, i - 1>(vec, vec[i - 1], std::forward<TValues>(values)...);
}

template <typename T, auto n>
[[nodiscard]] ARIA_HOST_DEVICE static constexpr auto ToTec(const Vec<T, n> &vec) {
  return ToTecImpl<T, n, n>(vec);
}

//
//
//
// Cast `Tec` to `Vec`.
template <typename T, typename... Ts>
[[nodiscard]] ARIA_HOST_DEVICE static constexpr auto ToVec(const Tec<T, Ts...> &tec) {
  using value_type = tup::detail::arithmetic_domain_t<T>;
  static_assert(tup::detail::is_tec_t_v<Tec<T, Ts...>, value_type>,
                "Element types of `Tec` should be \"as similar as possible\"");

  constexpr uint rank = rank_v<Tec<T, Ts...>>;

  Vec<value_type, rank> res;
  ForEach<rank>([&]<auto i>() { res[i] = get<i>(tec); });
  return res;
}

} // namespace vec::detail

} // namespace ARIA
