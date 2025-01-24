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
/// Before continue, define an operator:
///       +∞
/// ⟨x⟩ = ∫ x dξ.
///       -∞
///
/// We sometimes also want such kinds of integral domains:
/// ⟨x] = ∫(-∞, 0) x dξ
/// [x⟩ = ∫(0, +∞) x dξ
///
/// Moments of the distribution is defined as:
///   The 0^th order moment   : ⟨x⟩ = 1.
///   The 1^th order moment(s): ⟨ξx⟩ = u.
///   The 2^th order moment(s): ⟨ξξx⟩ = ...
///   The 3^th order moment(s): ⟨ξξξx⟩ = ...
///   ...
///
/// We can also define moments with ⟨x] or [x⟩.
template <uint dim, Real lambda>
class MaxwellBoltzmannDistribution;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/MaxwellBoltzmannDistribution.inc"
