#pragma once

/// \file
/// \warning This file is undergoing refinement,
/// interfaces are very unstable for now.

//
//
//
//
//
#include "ARIA/Constant.h"
#include "ARIA/ForEach.h"
#include "ARIA/Math.h"
#include "ARIA/MovingPoint.h"

#include <SmallVector.h>

namespace ARIA {

struct DegreeDynamic {};

template <uint v>
using Degree = C<v>;

struct NonRational {};

struct Rational {};

//
//
//
template <typename T, auto dim, typename RationalOrNot, typename TDegree, typename TControlPoints>
class BezierCurve;

//
//
//
template <typename T, auto dim, typename RationalOrNot, uint degree, typename TControlPoints>
class BezierCurve<T, dim, RationalOrNot, Degree<degree>, TControlPoints> {
public:
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
    llvm_vecsmall::SmallVector<VecCP, 10> temp(nCPs);

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
