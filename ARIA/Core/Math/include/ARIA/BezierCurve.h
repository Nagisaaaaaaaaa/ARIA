#pragma once

/// \file
/// \brief A policy-based Bezier curve implementation.
///
/// A Bezier curve is a parametric curve used in computer graphics and related fields.
/// A set of discrete "control points" defines a smooth, continuous curve by means of a formula.
/// Usually the curve is intended to approximate a real-world shape that otherwise
/// has no mathematical representation or whose representation is unknown or too complicated.
//
//
//
//
//
#include "ARIA/Curve.h"
#include "ARIA/ForEach.h"
#include "ARIA/Math.h"
#include "ARIA/SmallVector.h"

namespace ARIA {

/// \brief A policy-based Bezier curve implementation.
///
/// A Bezier curve is a parametric curve used in computer graphics and related fields.
/// A set of discrete "control points" defines a smooth, continuous curve by means of a formula.
/// Usually the curve is intended to approximate a real-world shape that otherwise
/// has no mathematical representation or whose representation is unknown or too complicated.
///
/// \tparam T Type to discretize the curve, you may want to use `float`, `double`, or `Real`.
/// \tparam dim Dimension of the `BezierCurve`.
/// \tparam TDegree Degree of the curve, can be determined at compile-time or runtime.
/// \tparam TControlPoints Type of the control points used to define the `BezierCurve`.
/// You can define owning `BezierCurve`s by entering:
/// `std::vector`, `std::array`, `thrust::host_vector`, `thrust::device_vector`, `TensorVector`, .etc.
/// You can also define non owning `BezierCurve` by entering:
/// `std::span`, `Tensor`, .etc.
///
/// \example ```cpp
/// // Define type of the control points.
/// using ControlPoints = std::vector<Vec3r>;
///
/// // Define type of the Bezier curve:
/// // 1. Use `Real` to discrete the curve.
/// // 2. Dimension equals to 3.
/// // 3. Degree is determined at compile-time, equals to 2.
/// //    Use `DegreeDynamic` for runtime determined degrees.
/// // 4. Specify type of the control points.
/// using Bezier = BezierCurve<Real, 3, Degree<2>, ControlPoints>;
///
/// // Create control points and the Bezier curve.
/// ControlPoints controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
/// Bezier bezier{controlPoints};
///
/// // We are to evaluate the curve at `t`.
/// Real t = 0.1_R;
///
/// // Whether `t` is in the domain of the curve (For any Bezier curves, in [0, 1]).
/// bool isInDomain = bezier.IsInDomain(t);
///
/// // Get position of the curve at `t`.
/// Vec3r position = bezier(t);
/// ```
template <typename T, auto dim, typename TDegree, typename TControlPoints>
class BezierCurve;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/BezierCurve.inc"
