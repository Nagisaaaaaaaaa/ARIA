#pragma once

namespace ARIA {

namespace boltzmann_distribution::detail {

// Statically test the template parameters for `Moment`.
template <uint dim, typename TOrder, typename TDomain, typename TU>
static consteval void StaticTestMoment() {
  static_assert(is_static_v<TOrder>, "The order type should be static");
  static_assert(is_static_v<TDomain>, "The domain type should be static");

  static_assert(tup::detail::is_tec_tr_v<TOrder, uint, dim>, "The order type should be `Tecu` and is `dim`-rank");
  static_assert(tup::detail::is_tec_tr_v<TDomain, int, dim>, "The domain type should be `Teci` and is `dim`-rank");
  static_assert(tup::detail::is_tec_tr_v<TU, Real, dim>, "The velocity type should be `Tecr` and is `dim`-rank");

  ForEach<dim>([]<auto i>() {
    using TDomainI = tup_elem_t<i, TDomain>;
    constexpr int domainI = TDomainI{};
    static_assert(domainI == -1 || domainI == 0 || domainI == 1, "Domain should only be -1, 0, or 1");
  });
}

//
//
//
// Get type of the sub-`Tup` with elements except the tail.
template <typename TTup>
struct except_tail {
  using TArray = to_type_array_t<TTup>;
  using TArrayExceptTail = TArray::template Slice<0, TArray::size - 1, 1>;
  using TTupExceptTail = to_tup_t<TArrayExceptTail>;
  using type = TTupExceptTail;
};

template <typename TTup>
using except_tail_t = typename except_tail<TTup>::type;

// Get type of the sub-`Tup` with only the tail element.
template <typename TTup>
struct tail {
  using TArray = to_type_array_t<TTup>;
  using TArrayTail = TArray::template Slice<TArray::size - 1, TArray::size, 1>;
  using TTupTail = to_tup_t<TArrayTail>;
  using type = TTupTail;
};

template <typename TTup>
using tail_t = typename tail<TTup>::type;

} // namespace boltzmann_distribution::detail

//
//
//
//
//
//! Mathematically, the n-D Boltzmann distribution has
//! good relationships with the 1D case, especially for moments computation.
//! So, template specialization will be used below.
template <Real lambda>
class BoltzmannDistribution<1, lambda> {
public:
  template <typename TOrder, typename TDomain, typename TU>
  [[nodiscard]] ARIA_HOST_DEVICE static constexpr Real Moment(const TU &u) {
    boltzmann_distribution::detail::StaticTestMoment<1, TOrder, TDomain, TU>();

    constexpr Real rt = 1.0 / (2.0 * lambda);

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
             ((order - 1) * rt) * Moment<decltype(TOrder{} - Tec<UInt<2>>{}), TDomain>(u);
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

    using TOrderL = boltzmann_distribution::detail::except_tail_t<TOrder>;
    using TOrderR = boltzmann_distribution::detail::tail_t<TOrder>;
    using TDomainL = boltzmann_distribution::detail::except_tail_t<TDomain>;
    using TDomainR = boltzmann_distribution::detail::tail_t<TDomain>;
    using TUL = boltzmann_distribution::detail::except_tail_t<TU>;
    using TUR = boltzmann_distribution::detail::tail_t<TU>;

    TUL uL;
    ForEach<dim - 1>([&]<auto i>() { get<i>(uL) = get<i>(u); });
    TUR uR;
    get<0>(uR) = get<dim - 1>(u);

    return BoltzmannDistribution<dim - 1, lambda>::template Moment<TOrderL, TDomainL, TUL>(uL) *
           BoltzmannDistribution<1, lambda>::template Moment<TOrderR, TDomainR, TUR>(uR);
  }
};

} // namespace ARIA
