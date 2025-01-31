#pragma once

/// \file
/// \brief Type set is a compile-time set containing unique types.
/// It has much higher perform than `TypeArray`.
/// TODO: Document this.

//
//
//
//
//
#include "ARIA/detail/TypeSetImpl.h"

namespace ARIA {

template <typename... Ts>
using MakeTypeSet = type_set::detail::to_type_set_t<MakeTypeArray<Ts...>>;

//
//
//
//
//
template <type_array::detail::NonArrayType... Ts>
  requires(type_set::detail::ValidTypeSetArgs<Ts...>)
struct TypeSet final : type_array::detail::TypeArrayBase {
private:
  using Idx = type_set::detail::Idx;
  using TArray = MakeTypeArray<Ts...>;
  using TNoCheck = type_set::detail::TypeSetNoCheck<Ts...>;

public:
  static constexpr size_t size = TArray::size;

  template <type_array::detail::NonArrayType T>
  static constexpr size_t nOf = TNoCheck::template has<T> ? 1 : 0;

  template <type_array::detail::NonArrayType T>
  static constexpr bool has = TNoCheck::template has<T>;

  template <type_array::detail::NonArrayType T>
  static constexpr size_t firstIdx = TNoCheck::template idx<T>;

  template <type_array::detail::NonArrayType T>
  static constexpr size_t lastIdx = TNoCheck::template idx<T>;

  template <type_array::detail::NonArrayType T>
  static constexpr size_t idx = TNoCheck::template idx<T>;

  //
  //
  //
  template <Idx i>
  using Get = TNoCheck::template Get<i>;

  // TODO: Tailored implementations needed for other features.
};

} // namespace ARIA
