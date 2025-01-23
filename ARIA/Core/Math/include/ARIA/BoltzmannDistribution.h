#pragma once

#include "ARIA/Tup.h"

namespace ARIA {

template <uint dim>
class BoltzmannDistribution;

template <>
class BoltzmannDistribution<1> {
public:
  template <typename TOrder, typename TDomain, typename TU>
  [[nodiscard]] ARIA_HOST_DEVICE static constexpr Real Moment(const TU &u) {
    static_assert(tup::detail::is_tec_tr_v<TOrder, uint, 1>, "The order type should be `Tec1u`");
    static_assert(tup::detail::is_tec_tr_v<TDomain, int, 1>, "The domain type should be `Tec1i`");
    static_assert(tup::detail::is_tec_tr_v<TU, Real, 1>, "The velocity type should be `Tec1r`");

    static_assert(is_static_v<TOrder>, "The order type should be static");
    static_assert(is_static_v<TDomain>, "The domain type should be static");

    constexpr uint order = get<0>(TOrder{});
    constexpr int domain = get<0>(TDomain{});

    static_assert(domain == -1 || domain == 0 || domain == 1, "Domain should only be -1, 0, or 1");
  }
};

} // namespace ARIA
