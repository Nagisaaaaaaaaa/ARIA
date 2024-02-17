#pragma once

#include "ARIA/Constant.h"
#include "ARIA/ForEach.h"
#include "ARIA/MovingPoint.h"

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
      auto round = _r + 1;
      ForEach<degree - round + 1>([&]<auto i>() { temp[i] = Lerp(temp[i], temp[i + 1], t); });
    });

    VecDim position;
    if constexpr (rational)
      position = temp[0].block<dim, 1>(0, 0) / temp[0][dim];
    else
      position = temp[0];
    return position;
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

public:
  ARIA_HOST_DEVICE explicit BezierCurve(const TControlPoints &controlPoints) : controlPoints_(controlPoints) {}

  ARIA_COPY_MOVE_ABILITY(BezierCurve, default, default);

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, controlPoints, controlPoints_);

public:
  [[nodiscard]] ARIA_HOST_DEVICE constexpr bool IsInDomain(const T &t) const { return T{0} <= t && t <= T{1}; }

  [[nodiscard]] ARIA_HOST_DEVICE VecDim operator()(const T &t) const {
    ARIA_ASSERT(IsInDomain(t));

    if (controlPoints().empty()) {
      return VecDim::Zero();
    }

    const uint nCPs = controlPoints().size();
    const uint degree = nCPs - 1;

    // Apply the deCasteljau algorithm, 1997, The NURBS Book, 24.
    std::vector<VecCP> temp(nCPs); // TODO: Optimize this line.

    for (uint i = 0; i < nCPs; ++i)
      temp[i] = controlPoints()[i];
    for (uint round = 1; round <= degree; ++round)
      for (uint i = 0; i <= degree - round; ++i)
        temp[i] = Lerp(temp[i], temp[i + 1], t);

    VecDim position;
    if constexpr (rational)
      position = temp[0].block<dim, 1>(0, 0) / temp[0][dim];
    else
      position = temp[0];
    return position;

    // TODO: Implement a GPU version.
  }

private:
  TControlPoints controlPoints_;
};

} // namespace ARIA
