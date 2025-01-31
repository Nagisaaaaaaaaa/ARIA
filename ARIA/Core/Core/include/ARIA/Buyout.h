#pragma once

#include "ARIA/detail/BuyoutImpl.h"

namespace ARIA {

template <typename F, typename... Ts>
using Buyout = buyout::detail::deduce_buyout_t<F, Ts...>;

//
//
//
template <typename... Ts>
ARIA_HOST_DEVICE static constexpr auto make_buyout(const auto &f) {
  using TBuyout = Buyout<std::decay_t<decltype(f)>, Ts...>;
  return TBuyout{f};
}

//
//
//
using buyout::detail::get;

} // namespace ARIA
