#pragma once

#include "ARIA/ARIA.h"

#include <concepts>

namespace ARIA {

/// \see is_invocable_with_brackets_v
template <typename T, typename... Ts>
struct is_invocable_with_brackets {
  static constexpr bool value = false;
};

// TODO: Update this since C++23.
template <typename T, typename... Ts>
  requires(sizeof...(Ts) == 1 &&
           requires(T t, Ts &&...ts) {
             { t[std::get<0>(std::forward_as_tuple(std::forward<Ts>(ts)...))] };
           })
struct is_invocable_with_brackets<T, Ts...> {
  static constexpr bool value = true;
};

/// \brief Whether expression `t[std::forward<Ts>(ts)...]` is valid
/// for an instance `t` of the given type `T`.
///
/// \example ```cpp
/// static_assert(is_invocable_with_brackets_v<std::vector<std::string>, int>);
/// ```
template <typename T, typename... Ts>
static constexpr bool is_invocable_with_brackets_v = is_invocable_with_brackets<T, Ts...>::value;

/// \brief Whether expression `t[std::forward<Ts>(ts)...]` is valid
/// for an instance `t` of the given type `T`.
///
/// \example ```cpp
/// static_assert(invocable_with_brackets<std::vector<std::string>, int>);
/// ```
template <typename T, typename... Ts>
concept invocable_with_brackets = is_invocable_with_brackets_v<T, Ts...>;

//
//
//
//
//
/// \see is_invocable_with_brackets_r_v
template <typename TRes, typename T, typename... Ts>
struct is_invocable_with_brackets_r {
  static constexpr bool value = false;
};

template <typename TRes, typename T, typename... Ts>
  requires(sizeof...(Ts) == 1 &&
           requires(T t, Ts &&...ts) {
             { t[std::get<0>(std::forward_as_tuple(std::forward<Ts>(ts)...))] } -> std::convertible_to<TRes>;
           })
struct is_invocable_with_brackets_r<TRes, T, Ts...> {
  static constexpr bool value = true;
};

/// \brief Whether expression `TRes r = t[std::forward<Ts>(ts)...]` is valid
/// for an instance `t` of the given type `T`.
///
/// \example ```cpp
/// static_assert(is_invocable_with_brackets_r_v<std::string, std::vector<std::string>, int>);
/// static_assert(is_invocable_with_brackets_r_v<double, std::vector<float>, int>);
/// ```
template <typename TRes, typename T, typename... Ts>
static constexpr bool is_invocable_with_brackets_r_v = is_invocable_with_brackets_r<TRes, T, Ts...>::value;

} // namespace ARIA
