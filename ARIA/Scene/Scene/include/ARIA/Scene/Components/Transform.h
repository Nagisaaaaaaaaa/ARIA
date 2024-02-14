#pragma once

#include "ARIA/Mat.h"
#include "ARIA/Quat.h"
#include "ARIA/Vec.h"
#include "ARIA/Scene/Component.h"
#include "ARIA/Scene/Object.h"

namespace ARIA {

/// \brief The coordinate space in which to operate.
///
/// See https://docs.unity3d.com/ScriptReference/Space.html.
enum class Space {
  Self,
  World,
};

//
//
//
//
//
/// \brief Position, rotation and scale of an object.
///
/// See https://docs.unity3d.com/ScriptReference/Transform.html.
///
/// \note ARIA uses radians for Euler angles, while Unity uses degrees.
class Transform final : public Component {
public:
  /// \brief The parent transform of the current transform.
  ///
  /// \example ```cpp
  /// Transform& parent = t.parent();
  /// ```
  ///
  /// \note If the current transform is a "root" transform, that is, `IsRoot()` returns true,
  /// this function will return reference to the transform of the "halo root" object.
  /// The "halo root" object is the parent object of all "root" objects.
  /// The "halo root" transform is the parent transform of all "root" transforms.
  ///
  /// The halo root object is introduced to make the hierarchy like a "tree".
  /// That is, the halo root is the actual tree root of the hierarchy tree.
  ///
  /// So, users should not modify anything about the halo root.
  /// Or there will be undefined behaviors.
  ARIA_REF_PROP(public, , parent, ARIA_PROP_IMPL(parent)());

  /// \brief Get the "root" transform of the current transform.
  /// See `parent` for more details.
  ///
  /// \example ```cpp
  /// Transform& root = t.root();
  /// ```
  ARIA_REF_PROP(public, , root, ARIA_PROP_IMPL(root)());

  //
  //
  //
  /// \brief Position of the transform relative to the parent transform.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, localPosition);

  /// \brief The rotation of the transform relative to the transform rotation of the parent.
  ARIA_PROP_PREFAB_QUAT(public, public, , Quatr, localRotation);

  /// \brief The scale of the transform relative to the object's parent.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, localScale);

  //
  //
  //
  /// \brief The rotation as Euler angles in radians relative to the parent transform's rotation.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, localEulerAngles);

  //
  //
  //
  /// \brief The up axis of the transform in local space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, localUp);

  /// \brief The down axis of the transform in local space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, localDown);

  /// \brief The forward axis of the transform in local space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, localForward);

  /// \brief The back axis of the transform in local space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, localBack);

  /// \brief The left axis of the transform in local space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, localLeft);

  /// \brief The right axis of the transform in local space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, localRight);

  //
  //
  //
  /// \brief Matrix that transforms a point from local space into parent space.
  ARIA_PROP_PREFAB_MAT(public, private, , Mat4r, localToParentMat);

  /// \brief Matrix that transforms a point from local space into world space.
  ARIA_PROP_PREFAB_MAT(public, private, , Mat4r, localToWorldMat);

  //
  //
  //
  /// \brief The world space position of the transform.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, position);

  /// \brief A Quaternion that stores the rotation of the transform in world space.
  ARIA_PROP_PREFAB_QUAT(public, public, , Quatr, rotation);

  /// \brief The global scale of the object.
  ///
  /// \todo Skew has not been supported by ARIA now.
  /// In theory, SVD decomposition should be used to take skew into account,
  /// but the singular values returned from `singularValues` have been sorted in descending order.
  /// And there's no way to solve this problem.
  ///
  /// ```cpp
  /// Eigen::JacobiSVD<Eigen::Matrix3<Real>> svd(linearPart, Eigen::ComputeFullU | Eigen::ComputeFullV);
  /// Vec3r singularValues = svd.singularValues(); // Has been sorted.
  /// ```
  ///
  /// That is way even Unity has to perform some magic approximation.
  ARIA_PROP_PREFAB_VEC(public, private, , Vec3r, lossyScale);

  //
  //
  //
  /// \brief The up axis of the transform in world space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, up);

  /// \brief The down axis of the transform in world space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, down);

  /// \brief The forward axis of the transform in world space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, forward);

  /// \brief The back axis of the transform in world space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, back);

  /// \brief The left axis of the transform in world space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, left);

  /// \brief The right axis of the transform in world space.
  ARIA_PROP_PREFAB_VEC(public, public, , Vec3r, right);

public:
  //
  //
  //
  /// \brief Whether the current transform is a "root" transform.
  /// See `parent` for more details.
  ///
  /// \example ```cpp
  /// bool isRoot = t.IsRoot();
  /// ```
  [[nodiscard]] bool IsRoot() const;

  //
  //
  //
  /// \brief Transforms position from local space to world space.
  Vec3r TransformPoint(const Vec3r &point);

  /// \brief Transforms the position (x, y, z) from local space to world space.
  Vec3r TransformPoint(const Real &x, const Real &y, const Real &z);

  /// \brief Transforms vector from local space to world space.
  Vec3r TransformVector(const Vec3r &vector);

  /// \brief Transforms vector (x, y, z) from local space to world space.
  Vec3r TransformVector(const Real &x, const Real &y, const Real &z);

  /// \brief Transforms direction from local space to world space.
  Vec3r TransformDirection(const Vec3r &direction);

  /// \brief Transforms direction (x, y, z) from local space to world space.
  Vec3r TransformDirection(const Real &x, const Real &y, const Real &z);

  //
  //
  //
  /// \brief Moves the transform in the direction and distance of translation.
  void Translate(const Vec3r &translation, Space relativeTo = Space::Self);

  /// \brief Moves the transform in the direction and distance of translation (x, y, z).
  void Translate(const Real &x, const Real &y, const Real &z, Space relativeTo = Space::Self);

  /// \brief Applies a rotation of `eulers.z` radians around the z-axis,
  /// `eulers.y` radians around the y-axis, and `eulers.x` radians around the x-axis (in that order).
  void Rotate(const Vec3r &eulers, Space relativeTo = Space::Self);

  /// \brief Applies a rotation of `zAngle` radians around the z-axis,
  /// `yAngle` radians around the y-axis, and `xAngle` radians around the x-axis (in that order).
  void Rotate(const Real &xAngle, const Real &yAngle, const Real &zAngle, Space relativeTo = Space::Self);

  /// \brief Rotates the transform about axis passing through point in world coordinates by angle radians.
  void RotateAround(const Vec3r &point, const Vec3r &axis, const Real &angle);

  //
  //
  //
  //
  //
public:
  //! ARIA use left-handed coordinate system, and the `up` direction is (0, 1, 0).
  // clang-format off
  static /* constexpr */ Vec3r Up()      { return { 0,  1,  0}; }
  static /* constexpr */ Vec3r Down()    { return { 0, -1,  0}; }
  static /* constexpr */ Vec3r Forward() { return { 0,  0, -1}; }
  static /* constexpr */ Vec3r Back()    { return { 0,  0,  1}; }
  static /* constexpr */ Vec3r Left()    { return {-1,  0,  0}; }
  static /* constexpr */ Vec3r Right()   { return { 1,  0,  0}; }

  // clang-format on

  //
  //
  //
  //
  //
private:
  friend Object;

  using Base = Component;
  using Base::Base;

public:
  Transform(const Transform &) = delete;
  Transform(Transform &&) noexcept = delete;
  Transform &operator=(const Transform &) = delete;
  Transform &operator=(Transform &&) noexcept = delete;
  ~Transform() final = default;

  //
  //
  //
  //
  //
private:
  Vec3r localPosition_{0, 0, 0};
  Quatr localRotation_{Quatr::Identity()};
  Vec3r localScale_{1, 1, 1};

  //
  //
  //
  //
  //
  //
  //
  //
  //
  [[nodiscard]] const Transform &ARIA_PROP_IMPL(parent)() const;
  [[nodiscard]] Transform &ARIA_PROP_IMPL(parent)();
  [[nodiscard]] const Transform &ARIA_PROP_IMPL(root)() const;
  [[nodiscard]] Transform &ARIA_PROP_IMPL(root)();

  //
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(localPosition)() const;
  void ARIA_PROP_IMPL(localPosition)(const Vec3r &value);
  [[nodiscard]] Quatr ARIA_PROP_IMPL(localRotation)() const;
  void ARIA_PROP_IMPL(localRotation)(const Quatr &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(localScale)() const;
  void ARIA_PROP_IMPL(localScale)(const Vec3r &value);

  //
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(localEulerAngles)() const;
  void ARIA_PROP_IMPL(localEulerAngles)(const Vec3r &value);

  //
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(localUp)() const;
  void ARIA_PROP_IMPL(localUp)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(localDown)() const;
  void ARIA_PROP_IMPL(localDown)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(localForward)() const;
  void ARIA_PROP_IMPL(localForward)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(localBack)() const;
  void ARIA_PROP_IMPL(localBack)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(localLeft)() const;
  void ARIA_PROP_IMPL(localLeft)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(localRight)() const;
  void ARIA_PROP_IMPL(localRight)(const Vec3r &value);

  //
  [[nodiscard]] Mat4r ARIA_PROP_IMPL(localToParentMat)() const;
  [[nodiscard]] Mat4r ARIA_PROP_IMPL(localToWorldMat)() const;

  //
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(position)() const;
  void ARIA_PROP_IMPL(position)(const Vec3r &value);
  [[nodiscard]] Quatr ARIA_PROP_IMPL(rotation)() const;
  void ARIA_PROP_IMPL(rotation)(const Quatr &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(lossyScale)() const;

  //
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(up)() const;
  void ARIA_PROP_IMPL(up)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(down)() const;
  void ARIA_PROP_IMPL(down)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(forward)() const;
  void ARIA_PROP_IMPL(forward)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(back)() const;
  void ARIA_PROP_IMPL(back)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(left)() const;
  void ARIA_PROP_IMPL(left)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_IMPL(right)() const;
  void ARIA_PROP_IMPL(right)(const Vec3r &value);

  //
  using Affine3r = Eigen::Transform<Real, 3, Eigen::Affine>;
  [[nodiscard]] Affine3r localToParentAffine() const;
  [[nodiscard]] Affine3r localToWorldAffine() const;
};

} // namespace ARIA
