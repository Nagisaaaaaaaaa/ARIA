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
/// Note that if homogeneous coordinate is used, for example, (x, y, z, w).
/// `dim` should still be `3` at this time.
/// \tparam RationalOrNot Whether the `BezierCurve` is rational or not.
/// For example `dim` equals to `3` and this parameter is set to `Rational`,
/// then homogeneous coordinate (x, y, z, w) will be used.
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
/// // 3. Non-rational.
/// // 4. Degree is determined at compile-time, equals to 2.
/// // 5. Specify type of the control points.
/// using Bezier = BezierCurve<Real, 3, NonRational, Degree<2>, ControlPoints>;
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
///
/// ```cpp
/// // Define type of the control points.
/// using ControlPoints = std::vector<Vec3r>;
///
/// // Define type of the Bezier curve:
/// // 1. Use `Real` to discrete the curve.
/// // 2. Dimension equals to 2.
/// // 3. Rational.
/// // 4. Degree is determined at runtime.
/// // 5. Specify type of the control points.
/// using Bezier = BezierCurve<Real, 2, Rational, DegreeDynamic, ControlPoints>;
///
/// // Create control points and the Bezier curve.
/// // Note, since the curve is defined as rational,
/// // The following control points are in homogeneous coordinate.
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
/// // Note, since the curve is defined as 2D,
/// // the evaluation result is `Vec2r`.
/// Vec2r position = bezier(t);
/// ```
template <typename T, auto dim, typename RationalOrNot, typename TDegree, typename TControlPoints>
class BezierCurve;

//
//
//
/// \brief Alias for non-rational `BezierCurve`s.
///
/// \example ```cpp
/// using ControlPoints = std::vector<Vec3r>;
/// using Bezier = BezierCurveNonRational<Real, 3, Degree<2>, ControlPoints>;
/// ```
template <typename T, auto dim, typename TDegree, typename TControlPoints>
using BezierCurveNonRational = BezierCurve<T, dim, NonRational, TDegree, TControlPoints>;

/// \brief Alias for rational `BezierCurve`s.
///
/// \example ```cpp
/// using ControlPoints = std::vector<Vec3r>;
/// using Bezier = BezierCurveRational<Real, 3, Degree<2>, ControlPoints>;
/// ```
template <typename T, auto dim, typename TDegree, typename TControlPoints>
using BezierCurveRational = BezierCurve<T, dim, Rational, TDegree, TControlPoints>;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/BezierCurve.inc"
