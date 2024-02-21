#pragma once

#include "ARIA/ARIA.h"

#include <concepts>

namespace ARIA {

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

template <typename T, typename... Ts>
static constexpr bool is_invocable_with_brackets_v = is_invocable_with_brackets<T, Ts...>::value;

template <typename T, typename... Ts>
concept invocable_with_brackets = is_invocable_with_brackets_v<T, Ts...>;

//
//
//
//
//
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

template <typename TRes, typename T, typename... Ts>
static constexpr bool is_invocable_with_brackets_r_v = is_invocable_with_brackets_r<TRes, T, Ts...>::value;

} // namespace ARIA
