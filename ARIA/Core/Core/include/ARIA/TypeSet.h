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
#include "ARIA/ForEach.h"

namespace ARIA {

struct TypeSetNonSame {};

struct TypeSetNonAmbiguous {};

//
//
//
namespace type_set::detail {

template <typename T>
struct Wrap {
  using type = T;
};

//
//
//
template <size_t i, typename... Ts>
struct OverloadingNA;

template <size_t i, typename T>
struct OverloadingNA<i, T> {
  static consteval Wrap<T> type(C<i>);
  static consteval C<i> value(Wrap<T>);
};

template <size_t i, typename T, typename... Ts>
struct OverloadingNA<i, T, Ts...> : OverloadingNA<i + 1, Ts...> {
  using OverloadingNA<i + 1, Ts...>::type;
  using OverloadingNA<i + 1, Ts...>::value;

  static consteval Wrap<T> type(C<i>);
  static consteval C<i> value(Wrap<T>);
};

//
//
//
template <typename... Ts>
class ImplNA {
private:
  using TOverloading = OverloadingNA<0, Ts...>;

  template <size_t i>
  using GetNoCheck = typename decltype(TOverloading::type(C<i>{}))::type;

  template <typename T>
  static constexpr size_t idx_no_check = decltype(TOverloading::value(std::declval<Wrap<T>>())){};

public:
  template <size_t i>
    requires(i == idx_no_check<GetNoCheck<i>>)
  using Get = GetNoCheck<i>;

  template <typename T>
    requires(std::is_same_v<T, GetNoCheck<idx_no_check<T>>>)
  static constexpr size_t idx = idx_no_check<T>;
};

//
//
//
template <typename... Ts>
struct ValidTypeSetImpl {
  using TImpl = ImplNA<Ts...>;
  static constexpr bool value = (std::is_same_v<Ts, TImpl::template Get<TImpl::template idx<Ts>>> && ...);
};

template <typename... Ts>
concept ValidTypeSet = ValidTypeSetImpl<Ts...>::value;

} // namespace type_set::detail

//
//
//
template <typename... Ts>
  requires type_set::detail::ValidTypeSet<Ts...>
class TypeSet {
private:
  using TImpl = type_set::detail::ImplNA<Ts...>;

public:
  template <size_t i>
  using Get = TImpl::template Get<i>;

  template <typename T>
  static constexpr size_t idx = TImpl::template idx<T>;
};

} // namespace ARIA
