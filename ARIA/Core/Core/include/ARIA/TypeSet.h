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
#include "ARIA/Tup.h"

namespace ARIA {

namespace type_set::detail {

template <size_t i, typename... Ts>
struct Impl;

template <size_t i, typename T>
struct Impl<i, T> {
  static consteval C<i> value(T);
  static consteval Tup<T> type(C<i>);
};

template <size_t i, typename T, typename... Ts>
struct Impl<i, T, Ts...> : Impl<i + 1, Ts...> {
  using Impl<i + 1, Ts...>::value;
  using Impl<i + 1, Ts...>::type;

  static consteval C<i> value(T);
  static consteval Tup<T> type(C<i>);
};

} // namespace type_set::detail

//
//
//
template <typename... Ts>
class TypeSet {
private:
  using TImpl = type_set::detail::Impl<0, Ts...>;

  template <typename T>
  static constexpr size_t idx_no_check = decltype(TImpl::value(std::declval<T>())){};

  template <size_t i>
  using GetNoCheck = tup_elem_t<0, decltype(TImpl::type(C<i>{}))>;

public:
  template <size_t i>
    requires(i == idx_no_check<GetNoCheck<i>>)
  using Get = GetNoCheck<i>;

  template <typename T>
    requires(std::is_same_v<T, GetNoCheck<idx_no_check<T>>>)
  static constexpr size_t idx = idx_no_check<T>;
};

} // namespace ARIA
