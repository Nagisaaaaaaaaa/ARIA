#pragma once

#include "ARIA/Constant.h"
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
template <typename T, auto dim, typename TDegree, typename RationalOrNot>
class BezierCurve;

//
//
//
template <typename T, auto dim, typename RationalOrNot>
class BezierCurve<T, dim, DegreeDynamic, RationalOrNot> {
public:
  bool IsInDomain(const T &t) const {
    // TODO: Implement this.
  }

  const Vec<T, dim> &operator()(const T &t) const {
    // TODO: Implement this.
  }

private:
};

//
//
//
template <typename T, auto dim, uint degree, typename RationalOrNot>
class BezierCurve<T, dim, Degree<degree>, RationalOrNot> {
public:
  bool IsInDomain(const T &t) const {
    // TODO: Implement this.
  }

  const Vec<T, dim> &operator()(const T &t) const {
    // TODO: Implement this.
  }

private:
};

//
//
//
//
//
#if 0
class RationalBezierCurve {
public:
  ARIA_PROP_BEGIN(public, public, , Vector<Vec4r> &, controlPoints)
  ARIA_PROP_FUNC(public, , ., empty)
  ARIA_PROP_FUNC(public, , ., begin)
  ARIA_PROP_FUNC(public, , ., end)
    ARIA_SUB_PROP(, size_t, size)
  ARIA_PROP_END

  ARIA_PROP(public, public, , uint, nCurveSamples)

  ARIA_PROP_BEGIN(public, private, , const Vector<Vec3r> &, curveVertices)
  ARIA_PROP_FUNC(public, , ., empty)
  ARIA_PROP_FUNC(public, , ., begin)
  ARIA_PROP_FUNC(public, , ., end)
    ARIA_SUB_PROP(, size_t, size)
  ARIA_PROP_END

private:
  [[nodiscard]] const auto &ARIA_PROP_IMPL(controlPoints)() const { return controlPoints_; }

  [[nodiscard]] auto &ARIA_PROP_IMPL(controlPoints)() { return controlPoints_; }

  void ARIA_PROP_IMPL(controlPoints)(const Vector<Vec4r> &value) {
    controlPoints_ = value;
    UpdateCurve();
  }

  [[nodiscard]] const auto &ARIA_PROP_IMPL(nCurveSamples)() const { return nCurveSamples_; }

  void ARIA_PROP_IMPL(nCurveSamples)(const uint &value) {
    nCurveSamples_ = value;
    UpdateCurve();
  }

  [[nodiscard]] const auto &ARIA_PROP_IMPL(curveVertices)() const { return curveVertices_; }

  // FIXME: MSVC bug.
  void ARIA_PROP_IMPL(curveVertices)(const Vector<Vec3r> &value) const { throw std::runtime_error("Unreachable code"); }

private:
  static constexpr uint nCurveSamplesDefault = 100;

  Vector<Vec4r> controlPoints_;
  uint nCurveSamples_ = nCurveSamplesDefault;
  Vector<Vec3r> curveVertices_;

private:
  void UpdateCurve() {
    if (controlPoints().empty()) {
      curveVertices_.clear();
      return;
    }

    const auto nControlPoints = controlPoints().size();
    const auto degree = nControlPoints - 1;

    curveVertices_.size() = nCurveSamples();

    // Apply the deCasteljau algorithm, 1997, The NURBS Book, 24.
  #pragma omp parallel
    {
      // Per-thread temp.
      Vector<Vec4r> temp(nControlPoints);

      // Compute position for each curve sample.
  #pragma omp for
      for (uint i_samples = 0; i_samples < nCurveSamples(); ++i_samples) {
        // TODO: use rip-map-like sample schemes
        Real u = static_cast<Real>(i_samples) / static_cast<Real>(nCurveSamples() - 1);

        for (uint i = 0; i < nControlPoints; ++i)
          temp[i] = controlPoints()[i];
        for (uint round = 1; round <= degree; ++round)
          for (uint i = 0; i <= degree - round; ++i)
            temp[i] = Lerp(temp[i], temp[i + 1], u);

        Vec3r position = Vec3r(temp[0].x(), temp[0].y(), temp[0].z()) / temp[0].w();
        curveVertices_[i_samples] = position;
      }
    }
  }
};
#endif

} // namespace ARIA
