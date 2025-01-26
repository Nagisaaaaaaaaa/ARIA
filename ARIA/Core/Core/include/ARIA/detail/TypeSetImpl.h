#pragma once

#include "ARIA/Constant.h"

namespace ARIA {

namespace type_set::detail {

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
