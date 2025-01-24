#pragma once

/// \file
/// \brief A policy-based Maxwell-Boltzmann distribution implementation.
//
//
//
//
//
#include "ARIA/Math.h"
#include "ARIA/Vec.h"

namespace ARIA {

template <uint dim, Real lambda>
class MaxwellBoltzmannDistribution;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/MaxwellBoltzmannDistribution.inc"
