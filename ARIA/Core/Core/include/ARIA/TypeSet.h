#pragma once

/// \file
/// \brief Type set is a compile-time 1D array containing any combinations of UNIQUE types
/// which themselves are not type arrays (we call them `NonArrayType`s).
///
/// That is, a type set can contain int, float&, void, etc.
/// But it cannot recursively contain another type array.
///
/// Type set may be helpful if you want to perform complex manipulations on many types,
/// for example, erase, remove, foreach, filter, etc.

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
