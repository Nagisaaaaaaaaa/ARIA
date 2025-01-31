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

  template <Idx begin, Idx end, Idx step>
  using Slice = MakeTypeSet<typename TArray::template Slice<begin, end, step>>;

  template <typename Void = void>
  using Reverse = MakeTypeSet<typename TArray::template Reverse<Void>>;

  template <Idx i>
  using Erase = MakeTypeSet<typename TArray::template Erase<i>>;

  template <Idx i, typename T>
  using Insert = MakeTypeSet<typename TArray::template Insert<i, T>>;

  template <typename Void = void>
  using PopFront = MakeTypeSet<typename TArray::template PopFront<Void>>;

  template <typename Void = void>
  using PopBack = MakeTypeSet<typename TArray::template PopBack<Void>>;

  template <typename T>
  using PushFront = MakeTypeSet<typename TArray::template PushFront<T>>;

  template <typename T>
  using PushBack = MakeTypeSet<typename TArray::template PushBack<T>>;

  template <type_array::detail::NonArrayType T>
  using Remove = MakeTypeSet<typename TArray::template Remove<T>>;

  template <type_array::detail::NonArrayType TFrom, typename TTo>
  using Replace = MakeTypeSet<typename TArray::template Replace<TFrom, TTo>>;

  //
  //
  //
  template <template <type_array::detail::NonArrayType> typename F>
  using ForEach = MakeTypeSet<typename TArray::template ForEach<F>>;

  template <template <type_array::detail::NonArrayType> typename F>
  using Filter = MakeTypeSet<typename TArray::template Filter<F>>;
};

} // namespace ARIA
