#pragma once

#include "ARIA/Tup.h"
#include "ARIA/TypeSet.h"

namespace ARIA {

namespace buyout::detail {

// Similar to `TensorVector`, for `Buyout`, we also want to write something like:
// 1. `Buyout<F, C<0>, C<1>, C<2>>`.
// 2. `Buyout<F, MakeTypeSet<C<0>, C<1>, C<2>>>`.
// 3. `Buyout<F, MakeTypeArray<C<0>, C<1>, C<2>>>`.
//
// All these 3 cases will be reduced to `BuyoutReduced<F, C<0>, C<1>, C<2>>`, which
// means that they will have exactly the same type.
template <typename F, type_array::detail::NonArrayType... TArgs>
class BuyoutReduced {
public:
  static_assert((std::is_same_v<TArgs, std::decay_t<TArgs>> && ...),
                "The buyout argument types should be decayed types");

private:
  //! Here, type duplications are automatically checked.
  using TArgsSet = MakeTypeSet<TArgs...>;
  //! The return types for every arguments are not required to be the same, so
  //! they will be stored in `Tup` instead of `std::array`.
  using TValuesTup = Tup<std::decay_t<decltype(std::declval<const F &>().template operator()<TArgs>())>...>;

public:
  ARIA_HOST_DEVICE constexpr explicit BuyoutReduced(const F &f) {
    ForEach<TArgsSet::size>([&]<auto i>() {
      using TArg = TArgsSet::template Get<i>;
      get<i>(values_) = f.template operator()<TArg>();
    });
  }

  ARIA_COPY_MOVE_ABILITY(BuyoutReduced, default, default);

public:
  template <typename TArg>
  ARIA_HOST_DEVICE constexpr decltype(auto) operator()() const {
    constexpr size_t i = TArgsSet::template idx<TArg>;
    return get<i>(values_);
  }

private:
  TValuesTup values_;
};

//
//
//
template <typename F, typename... Ts>
struct reduce_buyout;

template <typename F, type_array::detail::NonArrayType... Ts>
struct reduce_buyout<F, Ts...> {
  using type = BuyoutReduced<F, Ts...>;
};

template <typename F, template <typename...> typename T, type_array::detail::NonArrayType... Ts>
  requires(type_array::detail::ArrayType<T<Ts...>>)
struct reduce_buyout<F, T<Ts...>> {
  using type = BuyoutReduced<F, Ts...>;
};

template <typename F, typename... Ts>
using reduce_buyout_t = typename reduce_buyout<F, Ts...>::type;

//
//
//
template <typename TArg, typename F, type_array::detail::NonArrayType... TArgs>
ARIA_HOST_DEVICE static constexpr decltype(auto) get(const BuyoutReduced<F, TArgs...> &buyout) {
  return buyout.template operator()<TArg>();
}

} // namespace buyout::detail

} // namespace ARIA
