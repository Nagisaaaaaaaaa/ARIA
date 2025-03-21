#pragma once

#include "ARIA/TypeArray.h"

namespace ARIA {

namespace type_set::detail {

using Idx = type_array::detail::Idx;

//
//
//
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

//
//
//
//! C++ is something like a "weak typing" language.
//! For example, even though `int` and `const int` are different types,
//! both `void Func(int)` and `void Func(const int)` accept `int` and `const int`.
//!
//! To avoid such ambiguity, we have to "wrap" our types with
//! `void Func(Wrapper<int>)` and `void Func(Wrapper<const int>)`.
//! Note that `Wrapper<int>` and `Wrapper<const int>` can not be cast between each other.
//! Now, we successfully define a "strong typing" function `Func`.
template <typename T>
struct Wrapper {
  using type = T;
};

//
//
//
// Now we are ready to introduce the overloading magic.
//
// There are two overloaded functions here:
// 1. `Get`: Given the index in `[0, n - 1]`, returns the type in `Ts...`.
// 2. `idx`: Given the type in `Ts...`, returns the index in `[0, n - 1]`.
//
//! Note that the following `Get` and `idx` are only declared but not implemented, because
//! we don't have to implement them and, actually, there's no way to implement them.
//! 1. We only need to call them with `std::declval` and `decltype`.
//! 2. We are unable to call the constructors of `T` and return a real `Wrapper<T>` instance.
template <size_t i, typename... Ts>
struct Overloading;

// The grand class.
template <size_t i>
struct Overloading<i> {
  //! For the grand class, define the most generic `Get` which can accept all `C<...>`s.
  //! This is essential for defining empty `TypeSet`s.
  template <auto k>
  static consteval Wrapper<void> Get(C<k>);

  //! Also, the most generic `idx` is defined to accept all types.
  //! It returns a magic code which will be used to check duplications later.
  template <typename U>
  static consteval C<std::numeric_limits<size_t>::max()> idx(U);
};

// Recursive inheritance.
template <size_t i, typename T, typename... Ts>
struct Overloading<i, T, Ts...> : Overloading<i + 1, Ts...> {
  using Base = Overloading<i + 1, Ts...>;

  //! Call the parent's `idx` with the current type `T`.
  //! 1. If the magic code is returned, the most generic `idx` is called, which
  //!    means that `C<i> idx(Wrapper<T>)` has not been defined for the current `T`.
  //!    That is, `T` is not duplicated.
  //! 2. If the magic code is not returned, perform a similar analysis,
  //!    we can know that `T` must be duplicated.
  static_assert(decltype(Base::idx(std::declval<Wrapper<T>>())){} == std::numeric_limits<size_t>::max(),
                "Duplicated types are not allowed for `TypeSet`");

  //! Recursively using the parent's `Get` and `idx`.
  //! That's how function overloading is generated.
  using Base::Get;
  using Base::idx;

  static consteval Wrapper<T> Get(C<i>);
  static consteval C<i> idx(Wrapper<T>);
};

//
//
//
//
//
// Similar to `TypeArray`, negative indices should be supported.
// So we always need to convert indices to positive ones.
template <Idx i, size_t n>
struct pos_idx;

template <Idx i, size_t n>
  requires(i < 0)
struct pos_idx<i, n> {
  static constexpr size_t value = i + n;
};

template <Idx i, size_t n>
  requires(i >= 0)
struct pos_idx<i, n> {
  static constexpr size_t value = i;
};

template <Idx i, size_t n>
constexpr size_t pos_idx_v = pos_idx<i, n>::value;

//
//
//
// The `TypeSet` where duplications of `Ts...` have not been checked (`NoCheck`).
// Thanks to `Overloading`, the implementation is trivial.
// We only need to call `Overloading::Get` and `Overloading::idx`, then,
// the correct ones will be automatically chosen according to the function overloading rules.
template <typename... Ts>
class TypeSetNoCheck {
private:
  using TOverloading = Overloading<0, Ts...>;

  // Here, `no_check` means `T` may not be contained in `Ts...`.
  template <typename T>
  static constexpr size_t idx_no_check = decltype(TOverloading::idx(std::declval<Wrapper<T>>())){};

public:
  //! The magic code can also be used to implement `has`.
  template <typename T>
  static constexpr bool has = idx_no_check<T> != std::numeric_limits<size_t>::max();

  template <Idx i>
  using Get = typename decltype(TOverloading::Get(C<pos_idx_v<i, sizeof...(Ts)>>{}))::type;

  template <typename T>
    requires(has<T>)
  static constexpr size_t idx = idx_no_check<T>;
};

//
//
//
// The final step is to check duplications of `Ts...`.
//! We only need to call `Get` and `idx` with all possible combinations of arguments.
//! If there exists duplications, the code will be unable to compile.
template <typename... Ts>
struct ValidTypeSetArgsImpl {
  using TNoCheck = TypeSetNoCheck<Ts...>;
  static constexpr bool value =
      (std::is_same_v<Ts, typename TNoCheck::template Get<TNoCheck::template idx<Ts>>> && ...);
};

template <typename... Ts>
concept ValidTypeSetArgs = ValidTypeSetArgsImpl<Ts...>::value;

} // namespace type_set::detail

//
//
//
//
//
// fwd
template <type_array::detail::NonArrayType... Ts>
  requires(type_set::detail::ValidTypeSetArgs<Ts...>)
struct TypeSet;

//
//
//
namespace type_set::detail {

// Cast `TypeArray` to `TypeSet`.
template <typename... Ts>
consteval TypeSet<Ts...> ToTypeSetImpl(TypeArray<Ts...>) {
  return {};
}

template <typename T>
using to_type_set_t = decltype(ToTypeSetImpl(std::declval<T>()));

} // namespace type_set::detail

} // namespace ARIA
