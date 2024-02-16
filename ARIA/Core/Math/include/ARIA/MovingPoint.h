#pragma once

/// \file
/// \brief This file introduces the `MovingPoint` concept.
/// A moving point is defined as a callable variable.
/// Given any `t` satisfying `IsInDomain(t)`,
/// its `operator()(t)` returns the position at `t`.
///
/// `MovingPoint` the most basic concept satisfied by all the ARIA built-in curves.
//
//
//
//
//
#include "ARIA/Vec.h"

namespace ARIA {

/// \brief A moving point is defined as a callable variable.
/// Given any `t` satisfying `IsInDomain(t)`,
/// its `operator()(t)` returns the position at `t`.
///
/// \example ```cpp
/// template <typename T, auto dim, typename degree>
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
