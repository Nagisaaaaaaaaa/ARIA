#pragma once

/// \file
/// \brief TODO: Document this: SUPER CRAZY FAST compile-time type map.

//
//
//
//
//
#include "ARIA/Tup.h"

namespace ARIA {

namespace type_map::detail {

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

} // namespace type_map::detail

//
//
//
template <typename... Ts>
class TypeMap {
private:
  using TImpl = type_map::detail::Impl<0, Ts...>;

  template <typename T>
  static constexpr size_t find_no_check = decltype(TImpl::value(std::declval<T>())){};

  template <size_t i>
  using GetNoCheck = tup_elem_t<0, decltype(TImpl::type(C<i>{}))>;

public:
  template <size_t i>
    requires(i == find_no_check<GetNoCheck<i>>)
  using Get = GetNoCheck<i>;

  template <typename T>
    requires(std::is_same_v<T, GetNoCheck<find_no_check<T>>>)
  static constexpr size_t find = find_no_check<T>;
};

} // namespace ARIA
