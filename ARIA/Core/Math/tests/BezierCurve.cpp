#include "ARIA/BezierCurve.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(BezierCurve, Base) {
  static_assert(MovingPoint<BezierCurve, float, 1, DegreeDynamic, NonRational>);
  static_assert(MovingPoint<BezierCurve, double, 1, DegreeDynamic, NonRational>);
  static_assert(MovingPoint<BezierCurve, Real, 1, DegreeDynamic, NonRational>);

  static_assert(MovingPoint<BezierCurve, float, 1, Degree<2>, NonRational>);
  static_assert(MovingPoint<BezierCurve, double, 1, Degree<2>, NonRational>);
  static_assert(MovingPoint<BezierCurve, Real, 1, Degree<2>, NonRational>);
}

} // namespace ARIA
