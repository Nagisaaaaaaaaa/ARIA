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

//
//
//
template <typename T, auto dim, typename RationalOrNot, uint degree, typename TControlPoints>
class BezierCurve<T, dim, RationalOrNot, Degree<degree>, TControlPoints> {
public:
  using value_type = T;

  static constexpr bool rational = std::is_same_v<RationalOrNot, Rational>;

private:
  // `CP` is an abbreviation of "control point".
  using VecDim = Vec<T, dim>;
  using VecCP = std::conditional_t<rational, Vec<T, dim + 1>, VecDim>;
  static_assert(std::is_same_v<VecCP, typename TControlPoints::value_type>,
                "Type of control points does not match the dimension and rationality");
  //! Should not use `std::decay_t<decltype(std::declval<TControlPoints>()[0])>` in order to support proxy systems.

public:
  ARIA_HOST_DEVICE explicit BezierCurve(const TControlPoints &controlPoints) : controlPoints_(controlPoints) {}

  ARIA_COPY_MOVE_ABILITY(BezierCurve, default, default);

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, controlPoints, controlPoints_);

public:
  [[nodiscard]] ARIA_HOST_DEVICE constexpr bool IsInDomain(const T &t) const { return T{0} <= t && t <= T{1}; }

  [[nodiscard]] ARIA_HOST_DEVICE VecDim operator()(const T &t) const {
    ARIA_ASSERT(IsInDomain(t));

    // Apply the deCasteljau algorithm, 1997, The NURBS Book, 24.
    std::array<VecCP, nCPs> temp;

    ForEach<nCPs>([&]<auto i>() { temp[i] = controlPoints()[i]; });
    ForEach<degree>([&]<auto _r>() {
      constexpr auto round = _r + 1;
      ForEach<nCPs - round>([&]<auto i>() { temp[i] = Lerp(temp[i], temp[i + 1], t); });
    });

    VecDim pos;
    if constexpr (rational)
      pos = temp[0].head(dim) / temp[0][dim];
    else
      pos = temp[0];
    return pos;
  }

private:
  static constexpr uint nCPs = degree + 1;

  TControlPoints controlPoints_;
};

//
//
//
template <typename T, auto dim, typename RationalOrNot, typename TControlPoints>
class BezierCurve<T, dim, RationalOrNot, DegreeDynamic, TControlPoints> {
public:
  using value_type = T;

  static constexpr bool rational = std::is_same_v<RationalOrNot, Rational>;

private:
  // `CP` is an abbreviation of "control point".
  using VecDim = Vec<T, dim>;
  using VecCP = std::conditional_t<rational, Vec<T, dim + 1>, VecDim>;
  static_assert(std::is_same_v<VecCP, typename TControlPoints::value_type>,
                "Type of control points does not match the dimension and rationality");
  //! Should not use `std::decay_t<decltype(std::declval<TControlPoints>()[0])>` in order to support proxy systems.

public:
  ARIA_HOST_DEVICE explicit BezierCurve(const TControlPoints &controlPoints) : controlPoints_(controlPoints) {}

  ARIA_COPY_MOVE_ABILITY(BezierCurve, default, default);

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, controlPoints, controlPoints_);

public:
  [[nodiscard]] ARIA_HOST_DEVICE constexpr bool IsInDomain(const T &t) const { return T{0} <= t && t <= T{1}; }

  [[nodiscard]] ARIA_HOST_DEVICE VecDim operator()(const T &t) const {
    ARIA_ASSERT(IsInDomain(t));

    const uint nCPs = controlPoints().size();

#if ARIA_IS_HOST_CODE
    // Apply the deCasteljau algorithm, 1997, The NURBS Book, 24.
    small_vector<VecCP, 8> temp(nCPs); //! Local buffer optimization for degrees <= 7.

    for (uint i = 0; i < nCPs; ++i)
      temp[i] = controlPoints()[i];
    for (uint round = 1; round < nCPs; ++round)
      for (uint i = 0; i < nCPs - round; ++i)
        temp[i] = Lerp(temp[i], temp[i + 1], t);

    VecDim pos;
    if constexpr (rational)
      pos = temp[0].head(dim) / temp[0][dim];
    else
      pos = temp[0];
    return pos;
#else
    // Compute the combinations.
    // See https://stackoverflow.com/questions/11809502/which-is-better-way-to-calculate-ncr.
    auto nCr = [](uint n, uint r) {
      if (r > n - r)
        r = n - r; // Because C(n, r) == C(n, n - r).

      uint res = 1;
      for (uint i = 1; i <= r; i++) {
        res *= n - r + i;
        res /= i;
      }

      return res;
    };

    const uint degree = nCPs - 1;
    VecCP posHomo = VecCP::Zero();

    // Directly compute the Bernstein polynomials.
    for (uint i = 0; i < nCPs; ++i) {
      VecCP cp = controlPoints()[i];
      posHomo += cp * nCr(degree, i) * pow(1 - t, degree - i) * pow(t, i);
    }

    VecDim pos;
    if constexpr (rational)
      pos = posHomo.head(dim) / posHomo[dim];
    else
      pos = posHomo;
    return pos;
#endif
  }

private:
  TControlPoints controlPoints_;
};

} // namespace ARIA
