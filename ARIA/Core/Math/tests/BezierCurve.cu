#include "ARIA/BezierCurve.h"
#include "ARIA/Launcher.h"
#include "ARIA/TensorVector.h"

#include <gtest/gtest.h>
#include <span>

namespace ARIA {

namespace {

template <typename TBezier>
void ExpectSphereCUDA3D(const TBezier &bezier) {
  thrust::device_vector<int> successD(1);
  successD[0] = 1;

  Launcher(100, [=, success = successD.data()] ARIA_DEVICE(int i) {
    float t = static_cast<float>(i) / static_cast<float>(99);
    Vec3f pHomo = bezier(t);
    Vec2f p = Vec2f(pHomo.x(), pHomo.y()) / pHomo.z();
    if (std::abs(p.norm() - 1) > 0.0001F)
      success[0] = 0;
  }).Launch();

  cuda::device::current::get().synchronize();

  EXPECT_TRUE(successD[0] == 1);
}

template <typename TBezier>
void ExpectSphereCUDA2D(const TBezier &bezier) {
  thrust::device_vector<int> successD(1);
  successD[0] = 1;

  Launcher(100, [=, success = successD.data()] ARIA_DEVICE(int i) {
    float t = static_cast<float>(i) / static_cast<float>(99);
    Vec2f p = bezier(t);
    if (std::abs(p.norm() - 1) > 0.0001F)
      success[0] = 0;
  }).Launch();

  cuda::device::current::get().synchronize();

  EXPECT_TRUE(successD[0] == 1);
}

} // namespace

TEST(BezierCurve, Base) {
  static_assert(MovingPoint<BezierCurve, float, 1, DegreeDynamic, std::vector<Vec1f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, DegreeDynamic, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, DegreeDynamic, std::vector<Vec3f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, Degree<2>, std::vector<Vec1f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, Degree<2>, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, Degree<2>, std::vector<Vec3f>>);

  static_assert(MovingPoint<BezierCurve, float, 1, Degree<3>, std::vector<Vec1f>>);
  static_assert(MovingPoint<BezierCurve, float, 2, Degree<3>, std::vector<Vec2f>>);
  static_assert(MovingPoint<BezierCurve, float, 3, Degree<3>, std::vector<Vec3f>>);

  std::vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
  BezierCurve<float, 3, Degree<2>, std::vector<Vec3f>> bezier{controlPoints};

  static_assert(bezier.IsInDomain(0));
  static_assert(bezier.IsInDomain(1));
  static_assert(bezier.IsInDomain(0.1));
  static_assert(bezier.IsInDomain(0.9));
  static_assert(!bezier.IsInDomain(1.0001));
  static_assert(!bezier.IsInDomain(-0.0001));
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
    BezierCurve<float, 3, Degree<2>, std::vector<Vec3f>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Static degree + std::array.
  {
    std::array<Vec3f, 3> controlPoints = {Vec3f{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, Degree<2>, std::array<Vec3f, 3>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Static degree + std::span.
  {
    std::vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, Degree<2>, std::span<Vec3f>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Static degree + thrust::host_vector.
  {
    thrust::host_vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, Degree<2>, thrust::host_vector<Vec3f>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Static degree + thrust::device_vector.
  {
    thrust::device_vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, Degree<2>, thrust::device_vector<Vec3f>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Static degree + TensorVector / Tensor.
  {
    // Host + static.
    {
      auto controlPoints = make_tensor_vector<Vec3f, SpaceHost>(make_layout_major(C<3>{}));
      controlPoints[0] = {1, 0, 1};
      controlPoints[1] = {1, 1, 1};
      controlPoints[2] = {0, 2, 2};
      BezierCurve<float, 3, Degree<2>, decltype(controlPoints)> bezier0{controlPoints};
      expectSphere(bezier0);
      BezierCurve<float, 3, Degree<2>, std::decay_t<decltype(controlPoints.tensor())>> bezier1{controlPoints.tensor()};
      expectSphere(bezier1);
    }

    // Host + dynamic.
    {
      auto controlPoints = make_tensor_vector<Vec3f, SpaceHost>(make_layout_major(3));
      controlPoints[0] = {1, 0, 1};
      controlPoints[1] = {1, 1, 1};
      controlPoints[2] = {0, 2, 2};
      BezierCurve<float, 3, Degree<2>, decltype(controlPoints)> bezier0{controlPoints};
      expectSphere(bezier0);
      BezierCurve<float, 3, Degree<2>, std::decay_t<decltype(controlPoints.tensor())>> bezier1{controlPoints.tensor()};
      expectSphere(bezier1);
    }

    // Device + static.
    {
      auto controlPoints = make_tensor_vector<Vec3f, SpaceDevice>(make_layout_major(C<3>{}));
      controlPoints[0] = {1, 0, 1};
      controlPoints[1] = {1, 1, 1};
      controlPoints[2] = {0, 2, 2};
      BezierCurve<float, 3, Degree<2>, decltype(controlPoints)> bezier0{controlPoints};
      expectSphere(bezier0);
      BezierCurve<float, 3, Degree<2>, std::decay_t<decltype(controlPoints.tensor())>> bezier1{controlPoints.tensor()};
      expectSphere(bezier1);
      ExpectSphereCUDA3D(bezier1);
    }

    // Device + dynamic.
    {
      auto controlPoints = make_tensor_vector<Vec3f, SpaceDevice>(make_layout_major(3));
      controlPoints[0] = {1, 0, 1};
      controlPoints[1] = {1, 1, 1};
      controlPoints[2] = {0, 2, 2};
      BezierCurve<float, 3, Degree<2>, decltype(controlPoints)> bezier0{controlPoints};
      expectSphere(bezier0);
      BezierCurve<float, 3, Degree<2>, std::decay_t<decltype(controlPoints.tensor())>> bezier1{controlPoints.tensor()};
      expectSphere(bezier1);
      ExpectSphereCUDA3D(bezier1);
    }
  }

  // Dynamic degree + std::vector
  {
    std::vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, DegreeDynamic, std::vector<Vec3f>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Dynamic degree + std::array
  {
    std::array<Vec3f, 3> controlPoints = {Vec3f{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, DegreeDynamic, std::array<Vec3f, 3>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Dynamic degree + std::span
  {
    std::vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, DegreeDynamic, std::span<Vec3f>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Dynamic degree + thrust::host_vector.
  {
    thrust::host_vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, DegreeDynamic, thrust::host_vector<Vec3f>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Dynamic degree + thrust::device_vector.
  {
    thrust::device_vector<Vec3f> controlPoints = {{1, 0, 1}, {1, 1, 1}, {0, 2, 2}};
    BezierCurve<float, 3, DegreeDynamic, thrust::device_vector<Vec3f>> bezier{controlPoints};
    expectSphere(bezier);
  }

  // Dynamic degree + TensorVector / Tensor.
  {
    // Host + static.
    {
      auto controlPoints = make_tensor_vector<Vec3f, SpaceHost>(make_layout_major(C<3>{}));
      controlPoints[0] = {1, 0, 1};
      controlPoints[1] = {1, 1, 1};
      controlPoints[2] = {0, 2, 2};
      BezierCurve<float, 3, DegreeDynamic, decltype(controlPoints)> bezier0{controlPoints};
      expectSphere(bezier0);
      BezierCurve<float, 3, DegreeDynamic, std::decay_t<decltype(controlPoints.tensor())>> bezier1{
          controlPoints.tensor()};
      expectSphere(bezier1);
    }

    // Host + dynamic.
    {
      auto controlPoints = make_tensor_vector<Vec3f, SpaceHost>(make_layout_major(3));
      controlPoints[0] = {1, 0, 1};
      controlPoints[1] = {1, 1, 1};
      controlPoints[2] = {0, 2, 2};
      BezierCurve<float, 3, DegreeDynamic, decltype(controlPoints)> bezier0{controlPoints};
      expectSphere(bezier0);
      BezierCurve<float, 3, DegreeDynamic, std::decay_t<decltype(controlPoints.tensor())>> bezier1{
          controlPoints.tensor()};
      expectSphere(bezier1);
    }

    // Device + static.
    {
      auto controlPoints = make_tensor_vector<Vec3f, SpaceDevice>(make_layout_major(C<3>{}));
      controlPoints[0] = {1, 0, 1};
      controlPoints[1] = {1, 1, 1};
      controlPoints[2] = {0, 2, 2};
      BezierCurve<float, 3, DegreeDynamic, decltype(controlPoints)> bezier0{controlPoints};
      expectSphere(bezier0);
      BezierCurve<float, 3, DegreeDynamic, std::decay_t<decltype(controlPoints.tensor())>> bezier1{
          controlPoints.tensor()};
      expectSphere(bezier1);
      ExpectSphereCUDA3D(bezier1);
    }

    // Device + dynamic.
    {
      auto controlPoints = make_tensor_vector<Vec3f, SpaceDevice>(make_layout_major(3));
      controlPoints[0] = {1, 0, 1};
      controlPoints[1] = {1, 1, 1};
      controlPoints[2] = {0, 2, 2};
      BezierCurve<float, 3, DegreeDynamic, decltype(controlPoints)> bezier0{controlPoints};
      expectSphere(bezier0);
      BezierCurve<float, 3, DegreeDynamic, std::decay_t<decltype(controlPoints.tensor())>> bezier1{
          controlPoints.tensor()};
      expectSphere(bezier1);
      ExpectSphereCUDA3D(bezier1);
    }
  }
}

} // namespace ARIA
