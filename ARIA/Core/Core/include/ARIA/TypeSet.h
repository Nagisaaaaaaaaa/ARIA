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

template <type_array::detail::NonArrayType... Ts>
  requires(type_set::detail::ValidTypeSetArgs<Ts...>)
struct TypeSet final : public type_array::detail::TypeArrayBase {
private:
  using Idx = type_array::detail::Idx;
  using TArray = MakeTypeArray<Ts...>;
  using TNoCheck = type_set::detail::TypeSetNoCheck<Ts...>;

public:
  static constexpr size_t size = TArray::size;

  template <type_array::detail::NonArrayType T>
  static constexpr bool has = TNoCheck::template has<T>;

  template <size_t i>
  using Get = TNoCheck::template Get<i>;

  template <typename T>
  static constexpr size_t idx = TNoCheck::template idx<T>;
};

} // namespace ARIA
