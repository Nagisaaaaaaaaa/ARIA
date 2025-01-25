#pragma once

#include "ARIA/Tup.h"

namespace ARIA {

template <typename T, typename F, type_array::detail::NonArrayType... TArgs>
class Buyout {
private:
  using TArgsArray = MakeTypeArray<TArgs...>;
  using TValuesTup = Tup<std::decay_t<decltype(std::declval<F>().template operator()<TArgs>())>...>;

public:
  ARIA_HOST_DEVICE constexpr explicit Buyout(const F &f) {
    ForEach<TArgsArray::size>([&]<auto i>() {
      using TArg = TArgsArray::template Get<i>;
      get<i>(values_) = f.template operator()<TArg>();
    });
  }

  ARIA_COPY_MOVE_ABILITY(Buyout, default, default);

public:
  template <typename TArg>
  ARIA_HOST_DEVICE constexpr decltype(auto) operator()() const {
    constexpr size_t i = TArg2I<TArg>();
    return get<i>(values_);
  }

private:
  TValuesTup values_;

  template <typename TArg>
  static consteval size_t TArg2I() {
    // TODO: Type hash needed.
    size_t res;
    ForEach<TArgsArray::size>([&]<auto i>() {
      using TArg1 = TArgsArray::template Get<i>;
      if constexpr (std::is_same_v<TArg, TArg1>)
        res = i;
    });
    return res;
  }
};

template <typename TArg, typename T, typename F, type_array::detail::NonArrayType... TArgs>
ARIA_HOST_DEVICE static constexpr decltype(auto) get(const Buyout<T, F, TArgs...> &buyout) {
  return buyout.template operator()<TArg>();
}

} // namespace ARIA
