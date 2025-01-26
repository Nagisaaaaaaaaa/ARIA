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
#include "ARIA/detail/TypeSetImpl.h"

namespace ARIA {

template <typename... Ts>
  requires(type_set::detail::ValidTypeSet<Ts...>)
class TypeSet {
private:
  using TNoCheck = type_set::detail::TypeSetNoCheck<Ts...>;

public:
  template <size_t i>
  using Get = TNoCheck::template Get<i>;

  template <typename T>
  static constexpr size_t idx = TNoCheck::template idx<T>;
};

//
//
//
template <typename... Ts>
using MakeTypeSet = TypeSet<Ts...>;

} // namespace ARIA
