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
    static_assert(tup::detail::is_tec_tr_v<TOrder, uint, 1>, "The given order type should be `Tec1u`");
    static_assert(tup::detail::is_tec_tr_v<TDomain, int, 1>, "The given domain type should be `Tec1i`");
    static_assert(tup::detail::is_tec_tr_v<TU, Real, 1>, "The given velocity type should be `Tec1r`");
  }
};

} // namespace ARIA
