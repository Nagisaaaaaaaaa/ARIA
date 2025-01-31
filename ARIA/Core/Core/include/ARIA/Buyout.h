#pragma once

#include "ARIA/Tup.h"
#include "ARIA/TypeSet.h"

namespace ARIA {

template <typename F, type_array::detail::NonArrayType... TArgs>
class Buyout {
private:
  using TArgsSet = MakeTypeSet<TArgs...>;
  using TValuesTup = Tup<std::decay_t<decltype(std::declval<F>().template operator()<TArgs>())>...>;

public:
  ARIA_HOST_DEVICE constexpr explicit Buyout(const F &f) {
    ForEach<TArgsSet::size>([&]<auto i>() {
      using TArg = TArgsSet::template Get<i>;
      get<i>(values_) = f.template operator()<TArg>();
    });
  }

  ARIA_COPY_MOVE_ABILITY(Buyout, default, default);

public:
  template <typename TArg>
  ARIA_HOST_DEVICE constexpr decltype(auto) operator()() const {
    constexpr size_t i = TArgsSet::template idx<TArg>;
    return get<i>(values_);
  }

private:
  TValuesTup values_;
};

template <typename F, typename... Ts>
struct deduce_buyout;

template <typename F, type_array::detail::NonArrayType... Ts>
struct deduce_buyout<F, Ts...> {
  using type = Buyout<F, Ts...>;
};

template <typename F, template <typename...> typename T, type_array::detail::NonArrayType... Ts>
  requires(type_array::detail::ArrayType<T<Ts...>>)
struct deduce_buyout<F, T<Ts...>> {
  using type = Buyout<F, Ts...>;
};

template <typename F, typename... Ts>
using deduce_buyout_t = deduce_buyout<F, Ts...>::type;

template <typename... Ts>
ARIA_HOST_DEVICE static constexpr auto make_buyout(const auto &f) {
  using TBuyout = deduce_buyout_t<std::decay_t<decltype(f)>, Ts...>;
  return TBuyout{f};
}

template <typename TArg, typename F, type_array::detail::NonArrayType... TArgs>
ARIA_HOST_DEVICE static constexpr decltype(auto) get(const Buyout<F, TArgs...> &buyout) {
  return buyout.template operator()<TArg>();
}

} // namespace ARIA
