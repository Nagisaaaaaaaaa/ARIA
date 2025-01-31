#pragma once

/// \file
/// \brief Sometimes the same function may be called "multiple" times.
/// Here, "multiple" means that it may be called 0, 1, 2, ... times, and
/// the number is likely to be but not always > 0.
///
/// For example, there is a task where:
/// 1. For 95% cases, we need to do 10 things for 5 times (totally 10 * 5 things).
/// 2. For  5% cases, we need to do  9 things for 5 times (totally  9 * 5 things).
///
/// It's OK to just do it or use `[[likely]]`, but for performance-sensitive codes, it is
/// usually faster to always do 10 things and store the 10 values for later usages.
/// Especially for GPU codes, where `if-else` may slow down too much.
///
/// That's the motivation of introducing `Buyout`.
/// A `Buyout` is defined by calling a function with several arguments and
/// store the values for later usages.
/// Here, the arguments are required to be determined at compile-time, which
/// means that no runtime `if-else` is allowed at the construction stage.
///
/// Try `Buyout`, and sometimes, it will be faster.

//
//
//
//
//
#include "ARIA/detail/BuyoutImpl.h"

namespace ARIA {

/// \brief A `Buyout` is defined by calling a function with several arguments and
/// store the values for later usages.
/// Here, the arguments are required to be determined at compile-time, which
/// means that no runtime `if-else` is allowed at the construction stage.
///
/// \example ```cpp
/// struct SizeOf {
///   template <typename T>
///   constexpr size_t operator()() const {
///     return sizeof(T);
///   }
/// };
///
/// // Here, `buyout0`, `buyout1`, and `buyout2` will have the same type.
/// constexpr Buyout<SizeOf, float, double> buyout0{SizeOf{}};
/// constexpr Buyout<SizeOf, MakeTypeArray<float, double>> buyout1{SizeOf{}};
/// constexpr Buyout<SizeOf, MakeTypeSet<float, double>> buyout2{SizeOf{}};
///
/// static_assert(buyout0.operator()<float>() == 4);
/// static_assert(buyout0.operator()<double>() == 8);
/// static_assert(get<float>(buyout0) == 4);
/// static_assert(get<double>(buyout0) == 8);
/// ```
template <typename F, typename... Ts>
using Buyout = buyout::detail::reduce_buyout_t<F, Ts...>;

//
//
//
/// \brief
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
