#pragma once

#include "ARIA/ARIA.h"

#include <concepts>

namespace ARIA {

template <typename T, typename... Ts>
struct is_invocable_with_bracket {
  static constexpr bool value = false;
};

template <typename T, typename... Ts>
  requires(sizeof...(Ts) == 1 &&
           requires(T t, Ts &&...ts) {
             { t[std::get<0>(std::forward_as_tuple(std::forward<Ts>(ts)...))] };
           })
struct is_invocable_with_bracket<T, Ts...> {
  static constexpr bool value = true;
};

template <typename T, typename... Ts>
static constexpr bool is_invocable_with_bracket_v = is_invocable_with_bracket<T, Ts...>::value;

} // namespace ARIA
