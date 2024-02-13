#pragma once

#include "Renderer.h"

namespace ARIA {

/// \brief Renders triangle meshes inserted by the `TriMeshFilter`.
///
/// https://docs.unity3d.com/ScriptReference/MeshRenderer.html.
class TriMeshRenderer final : public Renderer {
public:
  //
  //
  //
  //
  //
private:
  friend Object;

  using Base = Renderer;
  using Base::Base;

public:
  TriMeshRenderer(const TriMeshRenderer &) = delete;
  TriMeshRenderer(TriMeshRenderer &&) noexcept = delete;
  TriMeshRenderer &operator=(const TriMeshRenderer &) = delete;
  TriMeshRenderer &operator=(TriMeshRenderer &&) noexcept = delete;
  ~TriMeshRenderer() final = default;

  //
  //
  //
  //
  //
private:
};

} // namespace ARIA
