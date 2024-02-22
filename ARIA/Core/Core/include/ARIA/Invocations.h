#pragma once

/// \file
/// \brief
//
//
//
//
//
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

//
//
//
//
//
/// \brief Calls `t[std::forward<Ts>(ts)...]` and gets the return value if exists.
///
/// \example ```cpp
/// EXPECT_FLOAT_EQ(invoke_with_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, 0), 1.1F);
/// ```
template <typename T, typename... Ts>
decltype(auto) invoke_with_brackets(T &&t, Ts &&...ts) {
  static_assert(invocable_with_brackets<T, Ts...>,
                "The given types should satisfy the `invocable_with_brackets` concepts");

  return t[std::get<0>(std::forward_as_tuple(std::forward<Ts>(ts)...))];
}

/// \brief If expression `t(std::forward<Ts>(ts)...)` is valid,
/// calls it and gets the return value if exists.
/// Else if expression `t[std::forward<Ts>(ts)...]` is valid,
/// calls it and gets the return value if exists.
///
/// \example ```cpp
/// std::vector<float> storage = {1.1F, 2.2F, 3.3F};
/// auto accessor = [&](size_t i) -> decltype(auto) { return storage[i]; };
///
/// EXPECT_FLOAT_EQ(invoke_with_parentheses_or_brackets(storage, 0), 1.1F);
/// EXPECT_FLOAT_EQ(invoke_with_parentheses_or_brackets(accessor, 1), 2.2F);
/// ```
///
/// \warning By definition, `operator()` has higher priority than `operator[]`.
/// Use `invoke_with_brackets_or_parentheses` instead if you want
/// `operator[]` has higher priority.
template <typename T, typename... Ts>
decltype(auto) invoke_with_parentheses_or_brackets(T &&t, Ts &&...ts) {
  if constexpr (std::invocable<T, Ts...>)
    return std::invoke(std::forward<T>(t), std::forward<Ts>(ts)...);
  else if constexpr (invocable_with_brackets<T, Ts...>)
    return invoke_with_brackets(std::forward<T>(t), std::forward<Ts>(ts)...);
  else
    ARIA_STATIC_ASSERT_FALSE("The given types should satisfy either `invocable` or `invocable_with_brackets`");
}

/// \brief If expression `t[std::forward<Ts>(ts)...]` is valid,
/// calls it and gets the return value if exists.
/// Else if expression `t(std::forward<Ts>(ts)...)` is valid,
/// calls it and gets the return value if exists.
///
/// \example ```cpp
/// std::vector<float> storage = {1.1F, 2.2F, 3.3F};
/// auto accessor = [&](size_t i) -> decltype(auto) { return storage[i]; };
///
/// EXPECT_FLOAT_EQ(invoke_with_parentheses_or_brackets(storage, 0), 1.1F);
/// EXPECT_FLOAT_EQ(invoke_with_parentheses_or_brackets(accessor, 1), 2.2F);
/// ```
///
/// \warning By definition, `operator[]` has higher priority than `operator()`.
/// Use `invoke_with_parentheses_or_brackets` instead if you want
/// `operator()` has higher priority.
template <typename T, typename... Ts>
decltype(auto) invoke_with_brackets_or_parentheses(T &&t, Ts &&...ts) {
  if constexpr (invocable_with_brackets<T, Ts...>)
    return invoke_with_brackets(std::forward<T>(t), std::forward<Ts>(ts)...);
  else if constexpr (std::invocable<T, Ts...>)
    return std::invoke(std::forward<T>(t), std::forward<Ts>(ts)...);
  else
    ARIA_STATIC_ASSERT_FALSE("The given types should satisfy either `invocable_with_brackets` or `invocable`");
}

} // namespace ARIA
