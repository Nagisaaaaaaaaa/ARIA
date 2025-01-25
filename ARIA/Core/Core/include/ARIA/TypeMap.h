#pragma once

#include "ARIA/Tup.h"

namespace ARIA {

namespace type_map::detail {

template <size_t i, typename... Ts>
struct FindImpl;

template <size_t i, typename T>
struct FindImpl<i, T> {
  static consteval C<i> value(T);
  static consteval Tup<T> type(C<i>);
};

template <size_t i, typename T, typename... Ts>
struct FindImpl<i, T, Ts...> : FindImpl<i + 1, Ts...> {
  using FindImpl<i + 1, Ts...>::value;
  using FindImpl<i + 1, Ts...>::type;

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
  using TArray = MakeTypeArray<Ts...>;
  using TFind = type_map::detail::FindImpl<0, Ts...>;

  template <typename T>
  static constexpr size_t find_no_check = decltype(TFind::value(std::declval<T>())){};

  template <size_t i>
  using GetNoCheck = tup_elem_t<0, decltype(TFind::type(C<i>{}))>;

public:
  template <size_t i>
    requires(i == find_no_check<GetNoCheck<i>>)
  using Get = GetNoCheck<i>;

  template <typename T>
    requires(std::is_same_v<T, GetNoCheck<find_no_check<T>>>)
  static constexpr size_t find = find_no_check<T>;
};

} // namespace ARIA
