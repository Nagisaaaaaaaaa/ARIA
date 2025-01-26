#pragma once

#include "ARIA/Constant.h"

namespace ARIA {

namespace type_set::detail {

//! For future developers: Please read the following comments to help you understand the codes.
//!
//! `TypeSet` is different from `TypeArray` in that the types are required to be unique.
//! For example, `TypeSet<int, float>` is OK, but `TypeSet<int, int>` is not allowed.
//! It seems to be a complex restriction, since we know that duplication check is at least O(n).
//! But for metaprogramming, this "restriction" allows us to implement the "fastest" `TypeSet`.
//! That's see how it works.
//!
//! Note that all the codes are executed by compilers.
//! One of the most important features for C++ compilers is to handle function overloading.
//! That is, given functions such as `void Func(int)`, `void Func(float)`, ...
//! The "best suited" `Func` should be chosen for any parameters.
//! So, suppose there are n candidates, we define
//!   O(Overload(n)): The time the compiler takes to handle function overloading.
//!
//! For all metaprogramming codes, this O(Overload(n)) can never be optimized, but
//! we can assume that O(Overload(n)) is relatively small,
//! (since the compilers never stuck for very long time, they are really strong, OvO).
//!
//! Let's see some examples about how our implementations can influence the compile time:
//! 1. `std::tuple_element_t<i, T>`:
//!      Takes O(n(Overload(n) + 1)) to compile because we should loop from `0` to `i`.
//! 2. `TypeArray::firstIdx<T>`:
//!      Takes O(n(Overload(n) + 1)) to compile because we should loop from the first type to `T`.
//!      Although same complexity, this feature is much slower than `std::tuple_element_t` because
//!      we need to compare the types with `std::is_same_v` at each iteration.
//!
//! So, the "fastest" `TypeSet` means that
//! all operations takes O(Overload(n) + 1) time to compile, which means that
//!   we have to make everything done at the overloading stage!
//!
//! This is impossible for `std::tuple` and `TypeArray`, where types may be duplicated.
//! But for `TypeSet`, where types are never duplicated, maybe... possible?
//!
//! Now, we've got the basic idea.

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
                    std::numeric_limits<size_t>::max(),
                "Duplicated types are not allowed for `TypeSet`");

  using Overloading<i + 1, Ts...>::type;
  using Overloading<i + 1, Ts...>::value;

  static consteval Wrap<T> type(C<i>);
  static consteval C<i> value(Wrap<T>);
};

//
//
//
template <typename... Ts>
class TypeSetNoCheck {
private:
  using TOverloading = Overloading<0, Ts...>;

public:
  template <size_t i>
  using Get = typename decltype(TOverloading::type(C<i>{}))::type;

  template <typename T>
  static constexpr size_t idx = decltype(TOverloading::value(std::declval<Wrap<T>>())){};
};

//
//
//
template <typename... Ts>
struct ValidTypeSetImpl {
  using TNoCheck = TypeSetNoCheck<Ts...>;
  static constexpr bool value = (std::is_same_v<Ts, TNoCheck::template Get<TNoCheck::template idx<Ts>>> && ...);
};

template <typename... Ts>
concept ValidTypeSet = ValidTypeSetImpl<Ts...>::value;

} // namespace type_set::detail

} // namespace ARIA
