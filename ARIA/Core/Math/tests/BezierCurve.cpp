#include "ARIA/BezierCurve.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(BezierCurve, Base) {
  static_assert(MovingPoint<BezierCurve, float, 1, NonRational, DegreeDynamic>);
  static_assert(MovingPoint<BezierCurve, double, 1, NonRational, DegreeDynamic>);
  static_assert(MovingPoint<BezierCurve, Real, 1, NonRational, DegreeDynamic>);

  static_assert(MovingPoint<BezierCurve, float, 1, NonRational, Degree<2>>);
  static_assert(MovingPoint<BezierCurve, double, 1, NonRational, Degree<2>>);
  static_assert(MovingPoint<BezierCurve, Real, 1, NonRational, Degree<2>>);

  BezierCurve<float, 2, NonRational, Degree<2>> b0;
  BezierCurve<float, 2, Rational, Degree<2>> b1;
  BezierCurve<float, 2, NonRational, DegreeDynamic> b2;
  BezierCurve<float, 2, Rational, DegreeDynamic> b3;
}

} // namespace ARIA
