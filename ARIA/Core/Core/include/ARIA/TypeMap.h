#pragma once

#include "ARIA/TypeArray.h"

namespace ARIA {

namespace type_map::detail {

template <size_t i, typename... Ts>
struct FindImpl;

template <size_t i, typename T>
struct FindImpl<i, T> {
  static consteval auto value(T) { return C<i>{}; }
};

template <size_t i, typename T, typename... Ts>
struct FindImpl<i, T, Ts...> : FindImpl<i + 1, Ts...> {
  using FindImpl<i + 1, Ts...>::value;

  static consteval auto value(T) { return C<i>{}; }
};

} // namespace type_map::detail

//
//
//
template <typename... Ts>
class TypeMap {
private:
  using TArray = MakeTypeArray<Ts...>;
  using TFind = type_map::detail::FindImpl<0, Ts...>;

public:
  template <size_t i>
  using Get = TArray::template Get<i>;

  template <typename T>
  static constexpr size_t find_unsafe = decltype(TFind::value(std::declval<T>())){};

  template <typename T>
  static constexpr size_t find = TArray::template firstIdx<T>;
};

} // namespace ARIA
