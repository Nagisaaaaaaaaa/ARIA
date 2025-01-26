#pragma once

/// \file
/// \brief TODO: Document this:
///              1. SUPER CRAZY FAST compile-time type set.
///              2. SUPER CRAZY STABLE!

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
  template <typename U>
  static consteval C<std::numeric_limits<size_t>::max()> value(U);

  static consteval Wrap<T> type(C<i>);
  static consteval C<i> value(Wrap<T>);
};

template <size_t i, typename T, typename... Ts>
struct Overloading<i, T, Ts...> : Overloading<i + 1, Ts...> {
  static_assert(decltype(Overloading<i + 1, Ts...>::value(std::declval<Wrap<T>>())){} ==
                std::numeric_limits<size_t>::max());

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
