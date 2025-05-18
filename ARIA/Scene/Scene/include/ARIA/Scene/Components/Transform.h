#pragma once

/// \file
/// \brief `Transform` is the position, rotation and scale of an `Object`.
///
/// `Transform` is implemented similar to Unity `Transform`,
/// see https://docs.unity3d.com/ScriptReference/Transform.html.
//
//
//
//
//
#include "ARIA/Scene/Component.h"
#include "ARIA/Scene/Components/detail/TransformImpl.h"

namespace ARIA {

/// \brief The coordinate space in which to operate.
///
/// `Space` has the same meaning with Unity `Space`,
/// see https://docs.unity3d.com/ScriptReference/Space.html.
enum class Space {
  Self,
  World,
};

//
//
//
//
//
/// \brief Position, rotation and scale of an `Object`.
///
/// `Transform` is implemented similar to Unity `Transform`,
/// see https://docs.unity3d.com/ScriptReference/Transform.html.
///
/// \example ```
/// Object& obj = Object::Create();
///
/// // Setup transform, there are plenty of properties, similar to Unity.
/// obj.transform().localPosition() = {1_R, 2_R, 3_R};
/// obj.transform().localRotation() = {1_R, 0_R, 0_R, 0_R};
/// obj.transform().localScale() = {1_R, 1_R, 1_R};
/// ```
///
/// \note `Transform` is a default and must component of any `Object`.
/// It is automatically added when the `Object` is created.
/// So, you do not need to, and are not allowed to
/// `obj.AddComponent<Transform>()` or `DestroyImmediate(obj.transform())`.
///
/// \warning ARIA uses radians for Euler angles, while Unity uses degrees.
class Transform final : public Component {
public:
  /// \brief The parent transform of the current transform.
  ///
  /// Please read the comments of `Object::parent` before continue.
  ///
  /// \example ```cpp
  /// Transform* parent = t.parent();
  ///
  /// Transform* newParent = ...;
  /// trans.parent() = newParent; // This will change the parent.
  /// parent = newParent;         //! WARNING, this will not work, see `Property.h` for the details.
  /// ```
  __ARIA_PROP_INCOMPLETE_PREFAB_TRANSFORM(public, public, , Transform *, parent);

  /// \brief Get the root transform of the current transform.
  ///
  /// Please read the comments of `Object::root` before continue.
  ///
  /// \example ```cpp
  /// Transform* root = obj.root();
  ///
  /// Transform* newRoot = ...;
  /// trans.root() = newRoot; // This will set parent of the original root object to `newRoot`.
  /// ```
  __ARIA_PROP_INCOMPLETE_PREFAB_TRANSFORM(public, public, , Transform *, root);

  //
  //
  //
  /// \brief Position of the transform relative to the parent transform.
  ///
  /// \example ```cpp
  /// Vec3r p = trans.localPosition();         // Get.
  /// trans.localPosition() = {0_R, 0_R, 0_R}; // Set.
  /// ```
  __ARIA_PROP_INCOMPLETE_PREFAB_VEC(public, public, , Vec3r, localPosition);

  /// \brief The rotation of the transform relative to the transform rotation of the parent.
  ///
  /// \example ```cpp
  /// Quatr r = trans.localRotation();              // Get.
  /// trans.localRotation() = {1_R, 0_R, 0_R, 0_R}; // Set.
  /// ```
  __ARIA_PROP_INCOMPLETE_PREFAB_QUAT(public, public, , Quatr, localRotation);

  /// \brief The scale of the transform relative to the object's parent.
  ///
  /// \example ```cpp
  /// Vec3r p = trans.localScale();         // Get.
  /// trans.localScale() = {1_R, 1_R, 1_R}; // Set.
  /// ```
  __ARIA_PROP_INCOMPLETE_PREFAB_VEC(public, public, , Vec3r, localScale);

  //
  //
  //
  /// \brief The rotation as Euler angles in radians relative to the parent transform's rotation.
  ARIA_PROP(public, public, , Vec3r, localEulerAngles);

  //
  //
  //
  /// \brief The up axis of the transform in local space.
  ARIA_PROP(public, public, , Vec3r, localUp);

  /// \brief The down axis of the transform in local space.
  ARIA_PROP(public, public, , Vec3r, localDown);

  /// \brief The forward axis of the transform in local space.
  ARIA_PROP(public, public, , Vec3r, localForward);

  /// \brief The back axis of the transform in local space.
  ARIA_PROP(public, public, , Vec3r, localBack);

  /// \brief The left axis of the transform in local space.
  ARIA_PROP(public, public, , Vec3r, localLeft);

  /// \brief The right axis of the transform in local space.
  ARIA_PROP(public, public, , Vec3r, localRight);

  //
  //
  //
  /// \brief Matrix that transforms a point from local space into parent space.
  ARIA_PROP(public, private, , Mat4r, localToParentMat);

  /// \brief Matrix that transforms a point from local space into world space.
  ARIA_PROP(public, private, , Mat4r, localToWorldMat);

  //
  //
  //
  /// \brief The world space position of the transform.
  __ARIA_PROP_INCOMPLETE_PREFAB_VEC(public, public, , Vec3r, position);

  /// \brief A Quaternion that stores the rotation of the transform in world space.
  __ARIA_PROP_INCOMPLETE_PREFAB_QUAT(public, public, , Quatr, rotation);

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
  ARIA_PROP(public, private, , Vec3r, lossyScale);

  //
  //
  //
  /// \brief The up axis of the transform in world space.
  ARIA_PROP(public, public, , Vec3r, up);

  /// \brief The down axis of the transform in world space.
  ARIA_PROP(public, public, , Vec3r, down);

  /// \brief The forward axis of the transform in world space.
  ARIA_PROP(public, public, , Vec3r, forward);

  /// \brief The back axis of the transform in world space.
  ARIA_PROP(public, public, , Vec3r, back);

  /// \brief The left axis of the transform in world space.
  ARIA_PROP(public, public, , Vec3r, left);

  /// \brief The right axis of the transform in world space.
  ARIA_PROP(public, public, , Vec3r, right);

  //
  //
  //
public:
  /// \brief Whether the current transform is a root transform.
  ///
  /// Please read the comments of `parent` before continue.
  ///
  /// \example ```cpp
  /// bool isRoot = trans.IsRoot();
  /// ```
  [[nodiscard]] bool IsRoot() const;

  /// \brief Is this transform a child (or a grandchild, or .etc) of `parent`?
  ///
  /// \example ```cpp
  /// bool isChildOf = trans.IsChildOf(anotherTrans);
  /// ```
  [[nodiscard]] bool IsChildOf(const Transform &parent) const;

  //
  //
  //
public:
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
private:
  friend Object;

  using Base = Component;

  /// \warning `Transform` is a default component of any `Object`.
  /// It is automatically added when the `Object` is created.
  /// So, you do not need to, and are not allowed to `obj.AddComponent<Transform>()`.
  using Base::Base;

public:
  ARIA_COPY_MOVE_ABILITY(Transform, delete, delete);

  /// \warning `Transform` is a must component of any `Object`.
  /// You are not allowed to `DestroyImmediate(obj.transform())`.
  ~Transform() final = default;

  //
  //
  //
public:
  /// \brief Two `Transform`s are defined as equal when they have exactly the same address.
  bool operator==(const Transform &other) const noexcept { return this == &other; }

  /// \brief Two `Transform`s are defined as equal when they have exactly the same address.
  bool operator!=(const Transform &other) const noexcept { return !operator==(other); }

  //
  //
  //
  //
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
  [[nodiscard]] const Transform *ARIA_PROP_GETTER(parent)() const;
  [[nodiscard]] Transform *ARIA_PROP_GETTER(parent)();
  void ARIA_PROP_SETTER(parent)(Transform *value);
  [[nodiscard]] const Transform *ARIA_PROP_GETTER(root)() const;
  [[nodiscard]] Transform *ARIA_PROP_GETTER(root)();
  void ARIA_PROP_SETTER(root)(Transform *value);

  //
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(localPosition)() const;
  void ARIA_PROP_SETTER(localPosition)(const Vec3r &value);
  [[nodiscard]] Quatr ARIA_PROP_GETTER(localRotation)() const;
  void ARIA_PROP_SETTER(localRotation)(const Quatr &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(localScale)() const;
  void ARIA_PROP_SETTER(localScale)(const Vec3r &value);

  //
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(localEulerAngles)() const;
  void ARIA_PROP_SETTER(localEulerAngles)(const Vec3r &value);

  //
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(localUp)() const;
  void ARIA_PROP_SETTER(localUp)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(localDown)() const;
  void ARIA_PROP_SETTER(localDown)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(localForward)() const;
  void ARIA_PROP_SETTER(localForward)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(localBack)() const;
  void ARIA_PROP_SETTER(localBack)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(localLeft)() const;
  void ARIA_PROP_SETTER(localLeft)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(localRight)() const;
  void ARIA_PROP_SETTER(localRight)(const Vec3r &value);

  //
  [[nodiscard]] Mat4r ARIA_PROP_GETTER(localToParentMat)() const;
  [[nodiscard]] Mat4r ARIA_PROP_GETTER(localToWorldMat)() const;

  //
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(position)() const;
  void ARIA_PROP_SETTER(position)(const Vec3r &value);
  [[nodiscard]] Quatr ARIA_PROP_GETTER(rotation)() const;
  void ARIA_PROP_SETTER(rotation)(const Quatr &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(lossyScale)() const;

  //
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(up)() const;
  void ARIA_PROP_SETTER(up)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(down)() const;
  void ARIA_PROP_SETTER(down)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(forward)() const;
  void ARIA_PROP_SETTER(forward)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(back)() const;
  void ARIA_PROP_SETTER(back)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(left)() const;
  void ARIA_PROP_SETTER(left)(const Vec3r &value);
  [[nodiscard]] Vec3r ARIA_PROP_GETTER(right)() const;
  void ARIA_PROP_SETTER(right)(const Vec3r &value);

  // Supporting methods.
  using Affine3r = Eigen::Transform<Real, 3, Eigen::Affine>;
  [[nodiscard]] Affine3r localToParentAffine() const;
  [[nodiscard]] Affine3r localToWorldAffine() const;
};

} // namespace ARIA
