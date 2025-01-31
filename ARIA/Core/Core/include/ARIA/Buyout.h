#pragma once

/// \file
/// \brief Sometimes the same function may be called "multiple" times.
/// Here, "multiple" means that it may be called 0, 1, 2, ... times, and
/// the number is likely to be but not always > 0.
///
/// It's OK to just call it, but for performance-sensitive codes, it is
/// usually faster to call it at least once and store the value for later usages.
/// Especially for GPU codes, where `if-else` may slow down too much.
///
/// That's the motivation of introducing `Buyout`.
/// A `Buyout` is defined by calling a function with several arguments and
/// store the values for later usages.
/// Here, the arguments are required to be determined at compile-time, which
/// means that no runtime `if-else` is allowed.

//
//
//
//
//
#include "ARIA/detail/BuyoutImpl.h"

namespace ARIA {

template <typename F, typename... Ts>
using Buyout = buyout::detail::reduce_buyout_t<F, Ts...>;

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
