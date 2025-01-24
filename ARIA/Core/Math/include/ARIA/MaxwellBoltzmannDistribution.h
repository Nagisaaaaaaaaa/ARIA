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
/// The distribution looks like f(ξ, u) = (λ / π)^(d / 2) exp(-λ(ξ - u)^2), where
/// ξ is the microscopic particle velocity,
/// u is the macroscopic velocity,
/// λ = 1 / (2 RT), R is the gas constant, T is the temperature,
/// d is the dimension of the system.
template <uint dim, Real lambda>
class MaxwellBoltzmannDistribution;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/MaxwellBoltzmannDistribution.inc"
