#pragma once

/// \file
/// \brief This file introduces the `MovingPoint` concept.
/// A moving point is defined as a function $f$ (in code, a callable variable `f`).
/// Given any function parameter $t$ (in code, variable `t`) which is
/// in the domain of $f$ (in code, `f.IsInDomain(t)` returns `true`),
/// $f(t)$ is the position of the point at $t$ (in code, `Vec<...> position = f(t)`).
///
/// It is named as "moving" point because $f(t)$ is moving by $t$.
/// For example, a Bezier curve is a point moving from $t = 0$ to $t = 1$.
/// When `0 <= t && t <= 1`, `bezier.IsInDomain(t)` returns `true`, and
/// `bezier(t)` returns the position of the curve at `t`.
///
/// `MovingPoint` is the most basic concept satisfied by all ARIA built-in curves.
//
//
//
//
//
#include "ARIA/Vec.h"

namespace ARIA {

/// \brief A moving point is defined as a function $f$ (in code, a callable variable `f`).
/// Given any function parameter $t$ (in code, variable `t`) which is
/// in the domain of $f$ (in code, `f.IsInDomain(t)` returns `true`),
/// $f(t)$ is the position of the point at $t$ (in code, `Vec<...> position = f(t)`).
///
/// It is named as "moving" point because $f(t)$ is moving by $t$.
/// For example, a Bezier curve is a point moving from $t = 0$ to $t = 1$.
/// When `0 <= t && t <= 1`, `bezier.IsInDomain(t)` returns `true`, and
/// `bezier(t)` returns the position of the curve at `t`.
///
/// `MovingPoint` is the most basic concept satisfied by all ARIA built-in curves.
///
/// \tparam TMovingPoint A template, whose template parameters will
/// be substituted with `TValue`, `dim`, and `TOthers...`.
/// \tparam dim Dimension of the moving point.
/// Note that if homogeneous coordinate is used, for example, (x, y, z, w).
/// `dim` should still be `3` at this time.
/// \tparam TOthers Other template parameters which have no relation with this concept.
///
/// \example ```cpp
/// template <typename T, auto dim, typename TDegree>
/// class BezierCurve {
/// public:
///   bool IsInDomain(const T &t) const { ... }
///
///   const Vec<T, dim> &operator()(const T &t) const { ... }
/// };
///
/// static_assert(MovingPoint<BezierCurve, Real, 2, C<5>>);
/// ```
///
/// \details This concept is named as `MovingPoint` instead of `Curve` because
/// we don't require that it should be continuous or smooth or not.
template <template <typename TValue, auto dim, typename... TOthers> typename TMovingPoint,
          typename TValue,
          auto dim,
          typename... TOthers>
concept MovingPoint =
    !std::integral<TValue> && requires(const TMovingPoint<TValue, dim, TOthers...> &movingPoint, const TValue &t) {
      { movingPoint.IsInDomain(t) } -> std::convertible_to<bool>;
      { movingPoint(t) } -> std::convertible_to<Vec<TValue, dim>>;
    };

} // namespace ARIA
