#include "ARIA/BezierCurve.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(BezierCurve, Base) {
  static_assert(MovingPoint<BezierCurve, float, 1, NonRational, DegreeDynamic, std::vector<Vec<float, 1>>>);
  static_assert(MovingPoint<BezierCurve, float, 2, NonRational, DegreeDynamic, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, NonRational, DegreeDynamic, std::vector<Vec3f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, Rational, DegreeDynamic, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, Rational, DegreeDynamic, std::vector<Vec3f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, Rational, DegreeDynamic, std::vector<Vec4f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, NonRational, Degree<2>, std::vector<Vec<float, 1>>>);
  static_assert(MovingPoint<BezierCurve, float, 2, NonRational, Degree<2>, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, NonRational, Degree<2>, std::vector<Vec3f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, Rational, Degree<2>, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, Rational, Degree<2>, std::vector<Vec3f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, Rational, Degree<2>, std::vector<Vec4f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, NonRational, Degree<3>, std::vector<Vec<float, 1>>>);
  static_assert(MovingPoint<BezierCurve, float, 2, NonRational, Degree<3>, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, NonRational, Degree<3>, std::vector<Vec3f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, Rational, Degree<3>, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, Rational, Degree<3>, std::vector<Vec3f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, Rational, Degree<3>, std::vector<Vec4f>>);

  // {
  //   std::vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 1, 2}};
  //   BezierCurve<float, 2, NonRational, Degree<2>, std::vector<Vec3f>> b{controlPoints};
  // }
}

} // namespace ARIA
