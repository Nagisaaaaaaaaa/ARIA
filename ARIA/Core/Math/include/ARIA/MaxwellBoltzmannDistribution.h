#pragma once

/// \file
/// \brief A policy-based Maxwell-Boltzmann distribution implementation.
/// See https://en.wikipedia.org/wiki/Maxwell-Boltzmann_distribution.
//
//
//
//
//
#include "ARIA/Math.h"
#include "ARIA/Vec.h"

namespace ARIA {

/// \brief A policy-based Maxwell-Boltzmann distribution implementation.
/// See https://en.wikipedia.org/wiki/Maxwell-Boltzmann_distribution.
///
/// The distribution looks like
/// f(ξ, u) = (λ / π)^(d / 2) exp(-λ(ξ - u)^2), where
///   ξ is the microscopic particle velocity,
///   u is the macroscopic velocity,
///   λ = 1 / (2 RT), R is the gas constant, T is the temperature,
///   d is the dimension of the system.
///
/// For simplicity, we firstly consider 1D systems.
/// Before continue, define an operator:
///       +∞
/// ⟨x⟩ = ∫ x dξ.
///       -∞
/// We also want other kinds of integral domains:
/// ⟨x] = ∫(-∞, 0) x dξ
/// [x⟩ = ∫(0, +∞) x dξ
///
/// Moments of the distribution are defined as:
///   The 0^th order moment: ⟨f⟩ = 1.
///   The 1^st order moment: ⟨ξf⟩ = u.
///   The 2^nd order moment: ⟨ξξf⟩ = ...
///   The 3^rd order moment: ⟨ξξξf⟩ = ...
///   ...
/// We can also define moments with ⟨x] or [x⟩.
///
/// Now, we move to 2D systems.
/// ξ = (ξ0, ξ1)^T and u = (u0, u1)^T are now vectors instead of scalars.
/// ⟨·⟩ should also be upgraded to
///         +∞+∞
/// ⟨⟨x⟩⟩ = ∫ ∫ x dξ0 dξ1.
///         -∞-∞
/// We can also define ⟨⟨x]⟩, ⟨⟨x]], ...
/// Note that there are now TWO 1^st order moments, ⟨⟨ξf⟩⟩ = u, and
/// THREE 2^nd order moments, ⟨⟨ξξf⟩⟩ = ....
/// We can also define moments with ⟨⟨x]⟩, ⟨⟨x]], ...
///
/// An important fact about high dimensional moments is that
/// ⟨⟨ ξ0^m ξ1^n f⟩⟩ = ⟨ξ0^m f⟩ ⟨ξ1^n f⟩.
/// This also works for other operators, which
/// means that n-D moments can always be computed based on 1D moments.
///
/// \tparam dim Dimension of the system.
/// \tparam lambda λ = 1 / (2 RT).
///
/// \example ```cpp
/// // 1D example.
/// using MBD = MaxwellBoltzmannDistribution<1, 1.5>;
///
/// using Order0 = Tec<UInt<0>>;
/// using Order1 = Tec<UInt<1>>;
/// using Order2 = Tec<UInt<2>>;
///
/// using DomainN = Tec<Int<-1>>;
/// using DomainO = Tec<Int<0>>;
/// using DomainP = Tec<Int<1>>;
///
/// Vec1r u{0.123_R}; // You can also use `Tec1r`.
///
/// Real o0dN = MBD::Moment<Order0, DomainN>(u); // ⟨f]
/// Real o1dO = MBD::Moment<Order1, DomainO>(u); // ⟨ξf⟩ = u
/// Real o2dP = MBD::Moment<Order2, DomainP>(u); // [ξξf⟩
/// ```
///
/// \example ```cpp
/// // 2D example.
/// using MBD = MaxwellBoltzmannDistribution<2, 1.5>;
///
/// using Order00 = Tec<UInt<0>, UInt<0>>;
/// using Order10 = Tec<UInt<1>, UInt<0>>;
/// using Order20 = Tec<UInt<2>, UInt<0>>;
/// using Order11 = Tec<UInt<1>, UInt<1>>;
///
/// using DomainNO = Tec<Int<-1>, Int<0>>;
/// using DomainOO = Tec<Int<0>, Int<0>>;
/// using DomainPO = Tec<Int<1>, Int<0>>;
/// using DomainPP = Tec<Int<1>, Int<1>>;
///
/// Vec2r u{0.123_R, 0.456_R}; // You can also use `Tec2r`.
///
/// Real o00dNO = MBD::Moment<Order00, DomainNO>(u); // ⟨⟨f]⟩ = ⟨f]⟨f⟩
/// Real o10dOO = MBD::Moment<Order10, DomainOO>(u); // ⟨⟨ξ0f⟩⟩ = ⟨ξ0f⟩⟨f⟩
/// Real o11dPO = MBD::Moment<Order11, DomainPO>(u); // ⟨[ξ0ξ1f⟩⟩ = [ξ0f⟩⟨ξ1f⟩
/// Real o02dPP = MBD::Moment<Order02, DomainPP>(u); // [[ξ1ξ1f⟩⟩ = [f⟩[ξ1ξ1f⟩
/// ```
template <uint dim, Real lambda>
class MaxwellBoltzmannDistribution;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/MaxwellBoltzmannDistribution.inc"
