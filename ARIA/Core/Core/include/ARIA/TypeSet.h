#pragma once

/// \file
/// \brief TODO: Document this:
///              1. SUPER CRAZY FAST compile-time type set.
///              2. Since the implementation is based on function overloading,
///                 any combinations of types which can potentially result in ambiguity is not allowed
///                 The rule is the same as function overloading:
///                   consteval size_t deduce(T0) { return 0; }
///                   consteval size_t deduce(T1) { return 0; }
///                 For example, duplications such as <int, int>, ambiguity such as <int, const int>.
///                 All these dangerous cases are checked by ARIA at compile-time, but
///                 you still need to pay much attensions.

//
//
//
//
//
#include "ARIA/Constant.h"

namespace ARIA {

namespace type_set::detail {

template <typename T>
struct Wrap {
  using type = T;
};

//
//
//
template <size_t i, typename... Ts>
struct Overloading;

template <size_t i, typename T>
struct Overloading<i, T> {
  static consteval Wrap<T> type(C<i>);
  static consteval C<i> value(Wrap<T>);
};

template <size_t i, typename T, typename... Ts>
struct Overloading<i, T, Ts...> : Overloading<i + 1, Ts...> {
  using Overloading<i + 1, Ts...>::type;
  using Overloading<i + 1, Ts...>::value;

  static consteval Wrap<T> type(C<i>);
  static consteval C<i> value(Wrap<T>);
};

} // namespace type_set::detail

//
//
//
template <typename... Ts>
class TypeSet {
private:
  using TOverloading = type_set::detail::Overloading<0, Ts...>;

public:
  template <size_t i>
  using Get = typename decltype(TOverloading::type(C<i>{}))::type;

  template <typename T>
  static constexpr size_t idx = decltype(TOverloading::value(std::declval<type_set::detail::Wrap<T>>())){};
};

} // namespace ARIA
