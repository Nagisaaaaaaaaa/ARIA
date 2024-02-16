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

  BezierCurve<float, 2, Degree<2>, NonRational> b0;
  BezierCurve<float, 2, Degree<2>, Rational> b1;
  BezierCurve<float, 2, DegreeDynamic, NonRational> b2;
  BezierCurve<float, 2, DegreeDynamic, Rational> b3;
}

} // namespace ARIA
