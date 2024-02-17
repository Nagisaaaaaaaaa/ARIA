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

//
//
//
/// \brief Whether the curve is non-rational.
///
/// \example ```cpp
/// // Define a non-rational `BezierCurve`.
/// BezierCurve<..., NonRational, ...> bezierCurve;
/// ```
struct NonRational {};

/// \brief Whether the curve is rational.
///
/// \example ```cpp
/// // Define a rational `BezierCurve`.
/// BezierCurve<..., Rational, ...> bezierCurve;
/// ```
struct Rational {};

} // namespace ARIA
