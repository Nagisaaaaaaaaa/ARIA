#include "ARIA/BezierCurve.h"

#include <gtest/gtest.h>

namespace ARIA {

TEST(BezierCurve, Base) {
  static_assert(MovingPoint<BezierCurve, float, 1, NonRational, DegreeDynamic, std::vector<Vec1f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, NonRational, DegreeDynamic, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, NonRational, DegreeDynamic, std::vector<Vec3f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, Rational, DegreeDynamic, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, Rational, DegreeDynamic, std::vector<Vec3f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, Rational, DegreeDynamic, std::vector<Vec4f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, NonRational, Degree<2>, std::vector<Vec1f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, NonRational, Degree<2>, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, NonRational, Degree<2>, std::vector<Vec3f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, Rational, Degree<2>, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, Rational, Degree<2>, std::vector<Vec3f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, Rational, Degree<2>, std::vector<Vec4f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, NonRational, Degree<3>, std::vector<Vec1f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, NonRational, Degree<3>, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, NonRational, Degree<3>, std::vector<Vec3f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, Rational, Degree<3>, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, Rational, Degree<3>, std::vector<Vec3f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, Rational, Degree<3>, std::vector<Vec4f>>);
}

TEST(BezierCurve, NonRational3D) {
  auto expectSphere = [](const auto &bezier) {
    for (float t = 0; t <= 1; t += 0.01) {
      Vec3f pHomo = bezier(t);
      Vec2f p = Vec2f(pHomo.x(), pHomo.y()) / pHomo.z();
      EXPECT_FLOAT_EQ(p.norm(), 1);
    }
  };

  // Static degree + std::vector.
  {
    std::vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, NonRational, Degree<2>, std::vector<Vec3f>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Static degree + std::array.
  {
    std::array<Vec3f, 3> controlPoints = {Vec3f{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, NonRational, Degree<2>, std::array<Vec3f, 3>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Dynamic degree + std::vector
  {
    std::vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, NonRational, DegreeDynamic, std::vector<Vec3f>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Dynamic degree + std::array
  {
    std::array<Vec3f, 3> controlPoints = {Vec3f{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, NonRational, DegreeDynamic, std::array<Vec3f, 3>> bezier{controlPoints};
    expectSphere(bezier);
  }
}

} // namespace ARIA
