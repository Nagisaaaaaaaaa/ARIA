#include "ARIA/Scene/Components/Camera.h"

namespace ARIA {

[[nodiscard]] bool Camera::ARIA_PROP_GETTER(perspective)() const {
  return !orthographic();
}

void Camera::ARIA_PROP_SETTER(perspective)(const bool &value) {
  orthographic() = !value;
}

//
//
//
//
//
Mat4r Camera::ARIA_PROP_GETTER(worldToCameraMat)() const {
  const Vec3r pos = transform().position();

  const Vec3r zAxis = transform().back().normalized();
  const Vec3r xAxis = transform().up().cross(zAxis).normalized();
  const Vec3r yAxis = zAxis.cross(xAxis);

  // clang-format off
  Mat4r viewMatrix;
  viewMatrix << xAxis.x(), xAxis.y(), xAxis.z(), -xAxis.dot(pos),
      yAxis.x(), yAxis.y(), yAxis.z(), -yAxis.dot(pos),
      zAxis.x(), zAxis.y(), zAxis.z(), zAxis.dot(pos),
      0, 0, 0, 1;
  // clang-format on

  return viewMatrix;
}

Mat4r Camera::ARIA_PROP_GETTER(projectionMat)() const {
  if (orthographic()) {
    Mat4r orthographicMatrix;
    return orthographicMatrix;
  } else { // Perspective.
    const Real tanHalfFOV = std::tan((fieldOfView() / 2_R) * (pi<Real> / 180_R));
    const Real farToNear = farClipPlane() - nearClipPlane();

    // clang-format off
    Mat4r perspectiveMatrix;
    perspectiveMatrix << 1_R / (aspect() * tanHalfFOV), 0, 0, 0,
        0, 1_R / (tanHalfFOV), 0, 0,
        0, 0, -(farClipPlane() + nearClipPlane()) / farToNear, -2 * farClipPlane() * nearClipPlane() / farToNear,
        0, 0, -1, 0;
    // clang-format on

    return perspectiveMatrix;
  }
}

} // namespace ARIA
