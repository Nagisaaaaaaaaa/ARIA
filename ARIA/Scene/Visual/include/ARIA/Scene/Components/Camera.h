#pragma once

#include "ARIA/Mat.h"
#include "ARIA/Scene/Behavior.h"
#include "ARIA/Scene/Components/Transform.h"

namespace ARIA {

/// \brief A `Camera` is a device through which the player views the world.
///
/// See https://docs.unity3d.com/ScriptReference/Camera.html.
///
/// \todo Support orthographic projection.
class Camera final : public Behavior {
public:
  /// \brief The color with which the screen will be cleared.
  ARIA_REF_PROP(public, , backgroundColor, backgroundColor_)

  //
  //
  //
  /// \brief Is the camera orthographic (true) or perspective (false).
  ARIA_REF_PROP(public, , orthographic, orthographic_)

  /// \brief Is the camera perspective (true) or orthographic (false).
  ARIA_PROP(public, public, , bool, perspective)

  /// \brief The vertical field of view of the `Camera`, in degrees.
  ARIA_REF_PROP(public, , fieldOfView, fieldOfView_)

  /// \brief The aspect ratio (width divided by height).
  ARIA_REF_PROP(public, , aspect, aspect_)

  /// \brief The distance of the near clipping plane from the the `Camera`, in world units.
  ARIA_REF_PROP(public, , nearClipPlane, nearClipPlane_)

  /// \brief The distance of the far clipping plane from the `Camera`, in world units.
  ARIA_REF_PROP(public, , farClipPlane, farClipPlane_)

  //
  //
  //
  /// \brief Matrix that transforms from world to camera space.
  /// This matrix is often referred to as "view matrix" in graphics literature.
  ARIA_PROP_PREFAB_MAT(public, private, , Mat4r, worldToCameraMat)

  /// \brief The perspective or orthographic matrix.
  ARIA_PROP_PREFAB_MAT(public, private, , Mat4r, projectionMat)

  //
  //
  //
  //
  //
private:
  friend Object;

  using Base = Behavior;
  using Base::Base;

public:
  Camera(const Camera &) = delete;
  Camera(Camera &&) noexcept = delete;
  Camera &operator=(const Camera &) = delete;
  Camera &operator=(Camera &&) noexcept = delete;
  ~Camera() final = default;

  //
  //
  //
  //
  //
private:
  Vec3r backgroundColor_{Vec3r::zero()};

  bool orthographic_{false};
  Real fieldOfView_{60_R};
  Real aspect_{1_R};
  Real nearClipPlane_{0.1_R};
  Real farClipPlane_{100_R};

  //
  //
  //
  //
  //
  //
  //
  //
  //
  [[nodiscard]] bool ARIA_PROP_IMPL(perspective)() const;
  void ARIA_PROP_IMPL(perspective)(const bool &value);

  //
  [[nodiscard]] Mat4r ARIA_PROP_IMPL(worldToCameraMat)() const;
  [[nodiscard]] Mat4r ARIA_PROP_IMPL(projectionMat)() const;
};

} // namespace ARIA
