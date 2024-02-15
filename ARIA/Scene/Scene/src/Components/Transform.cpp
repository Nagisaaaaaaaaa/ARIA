#include "ARIA/Scene/Components/Transform.h"

namespace ARIA {

const Transform *Transform::ARIA_PROP_IMPL(parent)() const {
  const Transform &t = object().parent()->transform();
  return &t;
}

Transform *Transform::ARIA_PROP_IMPL(parent)() {
  Transform &t = object().parent()->transform();
  return &t;
}

void Transform::ARIA_PROP_IMPL(parent)(Transform *value) {
  object().parent() = &value->object();
}

const Transform *Transform::ARIA_PROP_IMPL(root)() const {
  const Transform &t = object().root()->transform();
  return &t;
}

Transform *Transform::ARIA_PROP_IMPL(root)() {
  Transform &t = object().root()->transform();
  return &t;
}

void Transform::ARIA_PROP_IMPL(root)(Transform *value) {
  object().root() = &value->object();
}

//
//
//
//
//
Vec3r Transform::ARIA_PROP_IMPL(localPosition)() const {
  return localPosition_;
}

//! Only need to set `dirty` when any member variable is truly modified.
// TODO: It is non-trivial to implement dirty flags for `Transform`,
//       because when the parent `Transform` is set to dirty,
//       all its children should be recursively set to dirty.
void Transform::ARIA_PROP_IMPL(localPosition)(const Vec3r &value) {
  // dirty() = true;
  localPosition_ = value;
}

Quatr Transform::ARIA_PROP_IMPL(localRotation)() const {
  return localRotation_;
}

void Transform::ARIA_PROP_IMPL(localRotation)(const Quatr &value) {
  // dirty() = true;
  localRotation_ = value;
  localRotation_.normalize();
}

Vec3r Transform::ARIA_PROP_IMPL(localScale)() const {
  return localScale_;
}

void Transform::ARIA_PROP_IMPL(localScale)(const Vec3r &value) {
  // dirty() = true;
  localScale_ = value;
}

//
//
//
//
//
Vec3r Transform::ARIA_PROP_IMPL(localEulerAngles)() const {
  return ToEulerAngles(localRotation());
}

void Transform::ARIA_PROP_IMPL(localEulerAngles)(const Vec3r &value) {
  localRotation() = FromEulerAngles(value);
}

//
//
//
//
//
Vec3r Transform::ARIA_PROP_IMPL(localUp)() const {
  return localRotation() * Transform::Up();
}

void Transform::ARIA_PROP_IMPL(localUp)(const Vec3r &value) {
  // Compute the current up direction.
  const Vec3r curUp = localUp();

  // Compute the delta quaternion from the current up to the given up.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curUp, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = localRotation();
  localRotation() = rotDelta * curRot;
}

Vec3r Transform::ARIA_PROP_IMPL(localDown)() const {
  return localRotation() * Transform::Down();
}

void Transform::ARIA_PROP_IMPL(localDown)(const Vec3r &value) {
  // Compute the current down direction.
  const Vec3r curDown = localDown();

  // Compute the delta quaternion from the current down to the given down.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curDown, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = localRotation();
  localRotation() = rotDelta * curRot;
}

Vec3r Transform::ARIA_PROP_IMPL(localForward)() const {
  return localRotation() * Transform::Forward();
}

void Transform::ARIA_PROP_IMPL(localForward)(const Vec3r &value) {
  // Compute the current forward direction.
  const Vec3r curForward = localForward();

  // Compute the delta quaternion from the current forward to the given forward.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curForward, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = localRotation();
  localRotation() = rotDelta * curRot;
}

Vec3r Transform::ARIA_PROP_IMPL(localBack)() const {
  return localRotation() * Transform::Back();
}

void Transform::ARIA_PROP_IMPL(localBack)(const Vec3r &value) {
  // Compute the current back direction.
  const Vec3r curBack = localBack();

  // Compute the delta quaternion from the current back to the given back.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curBack, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = localRotation();
  localRotation() = rotDelta * curRot;
}

Vec3r Transform::ARIA_PROP_IMPL(localLeft)() const {
  return localRotation() * Transform::Left();
}

void Transform::ARIA_PROP_IMPL(localLeft)(const Vec3r &value) {
  // Compute the current left direction.
  const Vec3r curLeft = localLeft();

  // Compute the delta quaternion from the current left to the given left.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curLeft, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = localRotation();
  localRotation() = rotDelta * curRot;
}

Vec3r Transform::ARIA_PROP_IMPL(localRight)() const {
  return localRotation() * Transform::Right();
}

void Transform::ARIA_PROP_IMPL(localRight)(const Vec3r &value) {
  // Compute the current right direction.
  const Vec3r curRight = localRight();

  // Compute the delta quaternion from the current right to the given right.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curRight, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = localRotation();
  localRotation() = rotDelta * curRot;
}

//
//
//
//
//
Mat4r Transform::ARIA_PROP_IMPL(localToParentMat)() const {
  return localToParentAffine().matrix();
}

Mat4r Transform::ARIA_PROP_IMPL(localToWorldMat)() const {
  return localToWorldAffine().matrix();
}

//
//
//
//
//
Vec3r Transform::ARIA_PROP_IMPL(position)() const {
  if (IsRoot())
    return localPosition();
  else {
    const Transform *p = parent();
    return p->localToWorldAffine() * localPosition();
  }
}

void Transform::ARIA_PROP_IMPL(position)(const Vec3r &value) {
  if (IsRoot())
    localPosition() = value;
  else {
    const Transform *p = parent();
    localPosition() = p->localToWorldAffine().inverse() * value;
  }
}

Quatr Transform::ARIA_PROP_IMPL(rotation)() const {
  const Transform *p = this;

  Quatr res = localRotation();

  while (!p->IsRoot()) {
    p = p->parent();
    res = p->localRotation() * res;
  }

  return res.normalized();
}

void Transform::ARIA_PROP_IMPL(rotation)(const Quatr &value) {
  if (IsRoot())
    localRotation() = value;
  else {
    const Transform *p = parent();
    localRotation() = p->rotation().inverse() * value;
  }
}

Vec3r Transform::ARIA_PROP_IMPL(lossyScale)() const {
  Mat3r linearPart = localToWorldAffine().linear();

  Vec3r scales;
  scales.x() = linearPart.col(0).norm();
  scales.y() = linearPart.col(1).norm();
  scales.z() = linearPart.col(2).norm();

  return scales;
}

//
//
//
//
//
Vec3r Transform::ARIA_PROP_IMPL(up)() const {
  return rotation() * Transform::Up();
}

void Transform::ARIA_PROP_IMPL(up)(const Vec3r &value) {
  // Compute the current up direction.
  const Vec3r curUp = up();

  // Compute the delta quaternion from the current up to the given up.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curUp, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = rotation();
  rotation() = rotDelta * curRot;
}

Vec3r Transform::ARIA_PROP_IMPL(down)() const {
  return rotation() * Transform::Down();
}

void Transform::ARIA_PROP_IMPL(down)(const Vec3r &value) {
  // Compute the current down direction.
  const Vec3r curDown = down();

  // Compute the delta quaternion from the current down to the given down.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curDown, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = rotation();
  rotation() = rotDelta * curRot;
}

Vec3r Transform::ARIA_PROP_IMPL(forward)() const {
  return rotation() * Transform::Forward();
}

void Transform::ARIA_PROP_IMPL(forward)(const Vec3r &value) {
  // Compute the current forward direction.
  const Vec3r curForward = forward();

  // Compute the delta quaternion from the current forward to the given forward.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curForward, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = rotation();
  rotation() = rotDelta * curRot;
}

Vec3r Transform::ARIA_PROP_IMPL(back)() const {
  return rotation() * Transform::Back();
}

void Transform::ARIA_PROP_IMPL(back)(const Vec3r &value) {
  // Compute the current back direction.
  const Vec3r curBack = back();

  // Compute the delta quaternion from the current back to the given back.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curBack, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = rotation();
  rotation() = rotDelta * curRot;
}

Vec3r Transform::ARIA_PROP_IMPL(left)() const {
  return rotation() * Transform::Left();
}

void Transform::ARIA_PROP_IMPL(left)(const Vec3r &value) {
  // Compute the current left direction.
  const Vec3r curLeft = left();

  // Compute the delta quaternion from the current left to the given left.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curLeft, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = rotation();
  rotation() = rotDelta * curRot;
}

Vec3r Transform::ARIA_PROP_IMPL(right)() const {
  return rotation() * Transform::Right();
}

void Transform::ARIA_PROP_IMPL(right)(const Vec3r &value) {
  // Compute the current right direction.
  const Vec3r curRight = right();

  // Compute the delta quaternion from the current right to the given right.
  Quatr rotDelta;
  rotDelta.setFromTwoVectors(curRight, value.normalized());

  // Update rotation with the delta quaternion.
  const Quatr curRot = rotation();
  rotation() = rotDelta * curRot;
}

//
//
//
//
//
bool Transform::IsRoot() const {
  return object().IsRoot();
}

bool Transform::IsChildOf(const Transform &parent) const {
  return object().IsChildOf(parent.object());
}

//
//
//
//
//
Vec3r Transform::TransformPoint(const Vec3r &point) {
  return localToWorldAffine() * point.homogeneous();
}

Vec3r Transform::TransformPoint(const Real &x, const Real &y, const Real &z) {
  return TransformPoint({x, y, z});
}

Vec3r Transform::TransformVector(const Vec3r &vector) {
  return localToWorldAffine().linear() * vector;
}

Vec3r Transform::TransformVector(const Real &x, const Real &y, const Real &z) {
  return TransformVector({x, y, z});
}

Vec3r Transform::TransformDirection(const Vec3r &direction) {
  return rotation() * direction;
}

Vec3r Transform::TransformDirection(const Real &x, const Real &y, const Real &z) {
  return TransformDirection({x, y, z});
}

//
//
//
//
//
void Transform::Translate(const Vec3r &translation, Space relativeTo) {
  if (relativeTo == Space::Self) {
    // Convert translation to parent space using the local rotation.
    Vec3r translationParentSpace = localRotation() * translation;

    // Account for the parent's scale:
    // If we have a parent, we need to divide the translation by the parent's scale
    // because the translation in world space is affected by the scale of the parent.
    // If there is no parent, this step can be skipped.
    if (!IsRoot()) {
      const Transform *p = parent();
      const Vec3r parentScale = p->lossyScale();
      translationParentSpace = translationParentSpace.cwiseQuotient(parentScale);
    }

    // Update the local position.
    localPosition() += translationParentSpace;
  } else {
    position() += translation;
  }
}

void Transform::Translate(const Real &x, const Real &y, const Real &z, Space relativeTo) {
  Translate({x, y, z}, relativeTo);
}

void Transform::Rotate(const Vec3r &eulers, Space relativeTo) {
  // Create a rotation matrix from the Euler angles.
  const Eigen::AngleAxis<Real> rotationX(eulers.x(), Vec3r::UnitX());
  const Eigen::AngleAxis<Real> rotationY(eulers.y(), Vec3r::UnitY());
  const Eigen::AngleAxis<Real> rotationZ(eulers.z(), Vec3r::UnitZ());

  // Combine the rotations in the proper order (Z * Y * X).
  const Quatr deltaRotation = rotationZ * rotationY * rotationX;

  if (relativeTo == Space::Self) {
    // Apply transform in local space.
    localRotation() = localRotation() * deltaRotation;
  } else {
    // Apply transform in world space.
    rotation() = deltaRotation * rotation();
  }
}

void Transform::Rotate(const Real &xAngle, const Real &yAngle, const Real &zAngle, Space relativeTo) {
  Rotate({xAngle, yAngle, zAngle}, relativeTo);
}

void Transform::RotateAround(const Vec3r &point, const Vec3r &axis, const Real &angle) {
  // Create a rotation quaternion.
  const Quatr q = Quatr(Eigen::AngleAxisf(angle, axis.normalized()));
  // Compute the relative position.
  Vec3r relativePos = position() - point;
  // Rotate the relative position.
  relativePos = q * relativePos;
  // Update the world position.
  position() = point + relativePos;
  // Update the world rotation.
  rotation() = q * rotation();
}

//
//
//
//
//
Transform::Affine3r Transform::localToParentAffine() const {
  const Vec3r localP = localPosition();
  const Quatr localR = localRotation();
  const Vec3r localS = localScale();
  return Eigen::Translation3f(localP) * localR * Eigen::Scaling(localS);
}

Transform::Affine3r Transform::localToWorldAffine() const {
  const Transform *p = this;

  Affine3r res = localToParentAffine();

  while (!p->IsRoot()) {
    p = p->parent();
    res = p->localToParentAffine() * res;
  }

  return res;
}

} // namespace ARIA
