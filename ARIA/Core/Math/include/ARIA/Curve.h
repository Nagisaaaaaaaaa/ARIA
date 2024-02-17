#pragma once

/// \file
/// \brief This file defines several commonly used policies for
/// ARIA built-in curves, for example, `BezierCurve`.
//
//
//
//
//
#include "ARIA/Constant.h"
#include "ARIA/MovingPoint.h"

namespace ARIA {

/// \brief Degree of the curve, used to define curves whose
/// degree is determined at compile-time.
///
/// \example ```cpp
/// // Define a `BezierCurve` whose degree is 2.
/// BezierCurve<..., Degree<2>, ...> bezierCurve;
/// ```
template <uint v>
using Degree = C<v>;

/// \brief Degree of the curve, used to define curves whose
/// degree is determined at runtime (dynamic).
///
/// \example ```cpp
/// // Define a `BezierCurve` which has dynamic degree.
/// BezierCurve<..., DegreeDynamic, ...> bezierCurve;
/// ```
struct DegreeDynamic {};

} // namespace ARIA
