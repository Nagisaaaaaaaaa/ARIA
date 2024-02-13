#pragma once

#include "ARIA/Vec.h"
#include "Renderer.h"

#include <vector>

namespace ARIA {

/// \brief The line renderer is used to draw free-floating lines in 3D space.
///
/// See https://docs.unity3d.com/ScriptReference/LineRenderer.html.
class LineRenderer final : public Renderer {
public:
  ARIA_REF_PROP(public, , positions, positions_)

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
  LineRenderer(const LineRenderer &) = delete;
  LineRenderer(LineRenderer &&) noexcept = delete;
  LineRenderer &operator=(const LineRenderer &) = delete;
  LineRenderer &operator=(LineRenderer &&) noexcept = delete;
  ~LineRenderer() final = default;

  //
  //
  //
  //
  //
private:
  std::vector<Vec3r> positions_;
};

} // namespace ARIA
