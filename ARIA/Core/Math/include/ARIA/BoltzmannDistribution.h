#pragma once

#include "ARIA/Math.h"
#include "ARIA/Tup.h"
#include "ARIA/TypeArray.h"

namespace ARIA {

namespace boltzmann_distribution::detail {

template <uint dim, typename TOrder, typename TDomain, typename TU>
constexpr void StaticTestMoment() {
  static_assert(tup::detail::is_tec_tr_v<TOrder, uint, dim>, "The order type should be `Tecu`");
  static_assert(tup::detail::is_tec_tr_v<TDomain, int, dim>, "The domain type should be `Teci`");
  static_assert(tup::detail::is_tec_tr_v<TU, Real, dim>, "The velocity type should be `Tecr`");

  static_assert(is_static_v<TOrder>, "The order type should be static");
  static_assert(is_static_v<TDomain>, "The domain type should be static");

  ForEach<dim>([]<auto i>() {
    using TDomainI = tup_elem_t<i, TDomain>;
    constexpr int domainI = TDomainI{};
    static_assert(domainI == -1 || domainI == 0 || domainI == 1, "Domain should only be -1, 0, or 1");
  });
}

//
//
//
template <typename... Ts>
consteval auto ToTec(TypeArray<Ts...>) {
  return Tec<Ts...>{};
}

template <typename TArray>
using ToTec_t = decltype(ToTec(TArray{}));

template <typename... Ts>
consteval auto PopBack(Tec<Ts...>) {
  using TArray = MakeTypeArray<Ts...>;
  using TArrayBackPopped = TArray::template Slice<0, TArray::size - 1, 1>;
  return ToTec(TArrayBackPopped{});
}

template <typename TTec>
using PopBack_t = decltype(PopBack(TTec{}));

template <typename... Ts>
consteval auto Tail(Tec<Ts...>) {
  using TArray = MakeTypeArray<Ts...>;
  using TArrayTail = TArray::template Slice<TArray::size - 1, TArray::size, 1>;
  return ToTec(TArrayTail{});
}

template <typename TTec>
using Tail_t = decltype(Tail(TTec{}));

} // namespace boltzmann_distribution::detail

//
//
//
//
//
template <uint dim, Real lambda>
class BoltzmannDistribution;

//
//
//
template <Real lambda>
class BoltzmannDistribution<1, lambda> {
public:
  template <typename TOrder, typename TDomain, typename TU>
  [[nodiscard]] ARIA_HOST_DEVICE static constexpr Real Moment(const TU &u) {
    boltzmann_distribution::detail::StaticTestMoment<1, TOrder, TDomain, TU>();

    constexpr Real cs2 = 1.0 / (2.0 * lambda);

    constexpr uint order = get<0>(TOrder{});
    constexpr int domain = get<0>(TDomain{});

    Real u0 = get<0>(u);

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

//
//
//
template <uint dim, Real lambda>
  requires(dim > 1)
class BoltzmannDistribution<dim, lambda> {
public:
  template <typename TOrder, typename TDomain, typename TU>
  [[nodiscard]] ARIA_HOST_DEVICE static constexpr Real Moment(const TU &u) {
    boltzmann_distribution::detail::StaticTestMoment<dim, TOrder, TDomain, TU>();

    using TOrderL = boltzmann_distribution::detail::PopBack_t<TOrder>;
    using TOrderR = boltzmann_distribution::detail::Tail_t<TOrder>;
    using TDomainL = boltzmann_distribution::detail::PopBack_t<TDomain>;
    using TDomainR = boltzmann_distribution::detail::Tail_t<TDomain>;
    using TUL = boltzmann_distribution::detail::PopBack_t<TU>;
    using TUR = boltzmann_distribution::detail::Tail_t<TU>;

    TUL uL;
    ForEach<dim - 1>([&]<auto i>() { get<i>(uL) = get<i>(u); });
    TUR uR;
    get<0>(uR) = get<dim - 1>(u);

    return BoltzmannDistribution<dim - 1, lambda>::template Moment<TOrderL, TDomainL, TUL>(uL) *
           BoltzmannDistribution<1, lambda>::template Moment<TOrderR, TDomainR, TUR>(uR);
  }
};

} // namespace ARIA