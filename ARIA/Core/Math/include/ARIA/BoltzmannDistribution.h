#pragma once

#include "ARIA/Math.h"
#include "ARIA/Tup.h"

namespace ARIA {

template <uint dim, Real lambda>
class BoltzmannDistribution;

template <Real lambda>
class BoltzmannDistribution<1, lambda> {
public:
  template <typename TOrder, typename TDomain, typename TU>
  [[nodiscard]] ARIA_HOST_DEVICE static constexpr Real Moment(const TU &u) {
    constexpr Real cs2 = 1.0 / (2.0 * lambda);

    static_assert(tup::detail::is_tec_tr_v<TOrder, uint, 1>, "The order type should be `Tec1u`");
    static_assert(tup::detail::is_tec_tr_v<TDomain, int, 1>, "The domain type should be `Tec1i`");
    static_assert(tup::detail::is_tec_tr_v<TU, Real, 1>, "The velocity type should be `Tec1r`");

    static_assert(is_static_v<TOrder>, "The order type should be static");
    static_assert(is_static_v<TDomain>, "The domain type should be static");

    constexpr uint order = get<0>(TOrder{});
    constexpr int domain = get<0>(TDomain{});

    static_assert(domain == -1 || domain == 0 || domain == 1, "Domain should only be -1, 0, or 1");

    Real u0 = u[0];

    if constexpr (order == 0) {
      if constexpr (domain == 0)
        return 1_R;
      else if constexpr (domain == 1)
        return std::erfc(-std::sqrt(lambda) * u0) / 2_R;
      else if constexpr (domain == -1)
        return std::erfc(std::sqrt(lambda) * u0) / 2_R;
    } else if constexpr (order == 1) {
      if constexpr (domain == 0)
        return u0;
      else if constexpr (domain == 1)
        return u0 * Moment<Tec<UInt<0>>, TDomain>(u) +
               std::exp(-lambda * (u0 * u0)) / (std::sqrt(pi<Real> * lambda) * 2_R);
      else if constexpr (domain == -1)
        return u0 * Moment<Tec<UInt<0>>, TDomain>(u) -
               std::exp(-lambda * (u0 * u0)) / (std::sqrt(pi<Real> * lambda) * 2_R);
    } else {
      return u0 * Moment<decltype(TOrder{} - Tec<UInt<1>>{}), TDomain>(u) +
             ((order - 1) * cs2) * Moment<decltype(TOrder{} - Tec<UInt<2>>{}), TDomain>(u);
    }
  }
};

} // namespace ARIA
