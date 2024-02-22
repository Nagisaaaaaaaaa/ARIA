#pragma once

/// \file
/// \brief Suppose you want to implement a policy-based Bezier curve, where
/// type of the control points are determined by template parameters.
/// And you want to support both owning and non-owing control points.
/// Then, there are 2 simple cases:
/// 1. Type of the given control points is `std::vector<Vec3r>`.
///    This will create an owning Bezier curve.
/// 2. Type is `std::span<Vec3r>`, non-owning.
///
/// Only the above 2 cases? NO!
/// For GPU storages, data are usually stored as SOA instead of AOS.
/// For example, the control points may be stored in three `thrust::device_vector<Real>`s,
/// instead of one `thrust::device_vector<Vec3r>`.
///
/// In order to handle these kinds of storages, usually, accessors are introduced:
/// ```cpp
/// auto accessor = [&] (size_t i) {
///   return Vec3r{controlPointsX[i], controlPointsY[i], controlPointsZ[i]};
/// };
/// ```
/// Types of accessors (`decltype(accessor)`) will be given as template parameters,
/// instead of the underlying storages (`thrust::device_vector<Real>`).
/// But accessors usually use `operator()` instead of `operator[]`.
/// This makes it difficult to implement our generic Bezier curve.
///
/// So, we want a function, which automatically decides
/// whether to call `operator()` or `operator[]` and calls the correct one.
/// That is, we want to have more generic versions of `std::invoke` and `std::apply`,
/// which are able to consider also the `operator[]`s.
///
/// This file introduces such an implementation.
///
/// \todo Update all the implementations after C++23,
/// when `operator[]` can accept different numbers of parameters.
//
//
//
//
//
#include "ARIA/ARIA.h"

#include <cuda/std/functional>

namespace ARIA {

/// \see is_invocable_with_brackets_v
template <typename T, typename... Ts>
struct is_invocable_with_brackets {
  static constexpr bool value = false;
};

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
ARIA_HOST_DEVICE constexpr decltype(auto) invoke_with_brackets(T &&t, Ts &&...ts) {
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
ARIA_HOST_DEVICE constexpr decltype(auto) invoke_with_parentheses_or_brackets(T &&t, Ts &&...ts) {
  if constexpr (std::invocable<T, Ts...>)
    return cuda::std::invoke(std::forward<T>(t), std::forward<Ts>(ts)...); //! Should not use `std::invoke` instead.
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
ARIA_HOST_DEVICE constexpr decltype(auto) invoke_with_brackets_or_parentheses(T &&t, Ts &&...ts) {
  if constexpr (invocable_with_brackets<T, Ts...>)
    return invoke_with_brackets(std::forward<T>(t), std::forward<Ts>(ts)...);
  else if constexpr (std::invocable<T, Ts...>)
    return cuda::std::invoke(std::forward<T>(t), std::forward<Ts>(ts)...); //! Should not use `std::invoke` instead.
  else
    ARIA_STATIC_ASSERT_FALSE("The given types should satisfy either `invocable_with_brackets` or `invocable`");
}

//
//
//
//
//
namespace invocations::detail {

template <typename T, typename TTuple, std::size_t... I>
ARIA_HOST_DEVICE constexpr decltype(auto) apply_with_brackets_impl(T &&t, TTuple &&tuple, std::index_sequence<I...>) {
  //! Use `get` instead of `std::get` to support tuple types different from `std::tuple`,
  //! for example, `cuda::std::tuple`.
  //! Let ADL decides which `get` to call.
  return invoke_with_brackets(std::forward<T>(t), get<I>(std::forward<TTuple>(tuple))...);
}

template <typename T, typename TTuple, std::size_t... I>
ARIA_HOST_DEVICE constexpr decltype(auto)
apply_with_parentheses_or_brackets_impl(T &&t, TTuple &&tuple, std::index_sequence<I...>) {
  return invoke_with_parentheses_or_brackets(std::forward<T>(t), get<I>(std::forward<TTuple>(tuple))...);
}

template <typename T, typename TTuple, std::size_t... I>
ARIA_HOST_DEVICE constexpr decltype(auto)
apply_with_brackets_or_parentheses_impl(T &&t, TTuple &&tuple, std::index_sequence<I...>) {
  return invoke_with_brackets_or_parentheses(std::forward<T>(t), get<I>(std::forward<TTuple>(tuple))...);
}

} // namespace invocations::detail

//
//
//
/// \brief Calls `t[get<0>(tuple), get<1>(tuple), ...]` and gets the return value if exists.
///
/// \example ```cpp
/// EXPECT_FLOAT_EQ(apply_with_brackets(std::vector<float>{1.1F, 2.2F, 3.3F}, std::make_tuple(0)), 1.1F);
/// ```
template <typename T, typename TTuple>
ARIA_HOST_DEVICE constexpr decltype(auto) apply_with_brackets(T &&t, TTuple &&tuple) {
  return invocations::detail::apply_with_brackets_impl(
      std::forward<T>(t), std::forward<TTuple>(tuple),
      std::make_index_sequence<std::tuple_size_v<std::decay_t<TTuple>>>{});
}

/// \brief If expression `t(get<0>(tuple), get<1>(tuple), ...)` is valid,
/// calls it and gets the return value if exists.
/// Else if expression `t[get<0>(tuple), get<1>(tuple), ...]` is valid,
/// calls it and gets the return value if exists.
///
/// \example ```cpp
/// std::vector<float> storage = {1.1F, 2.2F, 3.3F};
/// auto accessor = [&](size_t i) -> decltype(auto) { return storage[i]; };
///
/// EXPECT_FLOAT_EQ(apply_with_parentheses_or_brackets(storage, std::make_tuple(0)), 1.1F);
/// EXPECT_FLOAT_EQ(apply_with_parentheses_or_brackets(accessor, std::make_tuple(1)), 2.2F);
/// ```
///
/// \warning By definition, `operator()` has higher priority than `operator[]`.
/// Use `apply_with_brackets_or_parentheses` instead if you want
/// `operator[]` has higher priority.
template <typename T, typename TTuple>
ARIA_HOST_DEVICE constexpr decltype(auto) apply_with_parentheses_or_brackets(T &&t, TTuple &&tuple) {
  return invocations::detail::apply_with_parentheses_or_brackets_impl(
      std::forward<T>(t), std::forward<TTuple>(tuple),
      std::make_index_sequence<std::tuple_size_v<std::decay_t<TTuple>>>{});
}

/// \brief If expression `t[get<0>(tuple), get<1>(tuple), ...]` is valid,
/// calls it and gets the return value if exists.
/// Else if expression `t(get<0>(tuple), get<1>(tuple), ...)` is valid,
/// calls it and gets the return value if exists.
///
/// \example ```cpp
/// std::vector<float> storage = {1.1F, 2.2F, 3.3F};
/// auto accessor = [&](size_t i) -> decltype(auto) { return storage[i]; };
///
/// EXPECT_FLOAT_EQ(apply_with_brackets_or_parentheses(storage, std::make_tuple(0)), 1.1F);
/// EXPECT_FLOAT_EQ(apply_with_brackets_or_parentheses(accessor, std::make_tuple(1)), 2.2F);
/// ```
///
/// \warning By definition, `operator[]` has higher priority than `operator()`.
/// Use `apply_with_brackets_or_parentheses` instead if you want
/// `operator()` has higher priority.
template <typename T, typename TTuple>
ARIA_HOST_DEVICE constexpr decltype(auto) apply_with_brackets_or_parentheses(T &&t, TTuple &&tuple) {
  return invocations::detail::apply_with_brackets_or_parentheses_impl(
      std::forward<T>(t), std::forward<TTuple>(tuple),
      std::make_index_sequence<std::tuple_size_v<std::decay_t<TTuple>>>{});
}

} // namespace ARIA
