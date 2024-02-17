#pragma once

#include "ARIA/Constant.h"
#include "ARIA/ForEach.h"
#include "ARIA/Math.h"
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
  static_assert(std::is_same_v<VecCP, std::decay_t<decltype(std::declval<TControlPoints>()[0])>>,
                "Type of control points does not match the dimension and rationality");

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
      ForEach<degree - round + 1>([&]<auto i>() { temp[i] = Lerp(temp[i], temp[i + 1], t); });
    });

    VecDim pos;
    if constexpr (rational)
      pos = temp[0].block<dim, 1>(0, 0) / temp[0][dim];
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
  static_assert(std::is_same_v<VecCP, std::decay_t<decltype(std::declval<TControlPoints>()[0])>>,
                "Type of control points does not match the dimension and rationality");

public:
  ARIA_HOST_DEVICE explicit BezierCurve(const TControlPoints &controlPoints) : controlPoints_(controlPoints) {}

  ARIA_COPY_MOVE_ABILITY(BezierCurve, default, default);

public:
  ARIA_REF_PROP(public, ARIA_HOST_DEVICE, controlPoints, controlPoints_);

public:
  [[nodiscard]] ARIA_HOST_DEVICE constexpr bool IsInDomain(const T &t) const { return T{0} <= t && t <= T{1}; }

  [[nodiscard]] ARIA_HOST_DEVICE VecDim operator()(const T &t) const {
#if ARIA_IS_HOST_CODE
    ARIA_ASSERT(IsInDomain(t));

    const uint nCPs = controlPoints().size();
    const uint degree = nCPs - 1;

    // Apply the deCasteljau algorithm, 1997, The NURBS Book, 24.
    std::vector<VecCP> temp(nCPs); // TODO: Optimize this line.

    for (uint i = 0; i < nCPs; ++i)
      temp[i] = controlPoints()[i];
    for (uint round = 1; round <= degree; ++round)
      for (uint i = 0; i <= degree - round; ++i)
        temp[i] = Lerp(temp[i], temp[i + 1], t);

    VecDim pos;
    if constexpr (rational)
      pos = temp[0].block<dim, 1>(0, 0) / temp[0][dim];
    else
      pos = temp[0];
    return pos;
#else
    VecCP posHomo = VecCP::Zero();

    for (uint i = 0; i <= degree; ++i) {
      T blend = T{1};
      T oneMinusT = T{1} - t;

      for (uint j = 0; j < i; ++j) {
        blend *= oneMinusT;
      }
      for (uint j = i + 1; j <= degree; ++j) {
        blend *= t;
      }

      posHomo += controlPoints()[i] * blend;
    }

    VecDim pos;
    if constexpr (rational)
      pos = posHomo.block<dim, 1>(0, 0) / posHomo[dim];
    else
      pos = posHomo;
    return pos;
#endif
  }

private:
  TControlPoints controlPoints_;
};

} // namespace ARIA
