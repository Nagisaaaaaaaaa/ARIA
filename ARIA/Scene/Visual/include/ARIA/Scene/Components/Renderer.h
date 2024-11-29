#pragma once

#include "ARIA/Scene/Component.h"
#include "ARIA/Scene/Material.h"

namespace ARIA {

/// \brief General functionality for all renderers.
///
/// A renderer is what makes an object appear on the screen.
///
/// See https://docs.unity3d.com/ScriptReference/Renderer.html.
class Renderer : public Component {
public:
  /// \brief Returns all the instantiated materials of this object.
  ARIA_REF_PROP(public, , materials, materials_)

  /// \brief Returns the first instantiated material assigned to the renderer.
  ARIA_PROP(public, private, , Material *, material)

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
  Renderer(const Renderer &) = delete;
  Renderer(Renderer &&) noexcept = delete;
  Renderer &operator=(const Renderer &) = delete;
  Renderer &operator=(Renderer &&) noexcept = delete;
  ~Renderer() override = default;

  //
  //
  //
  //
  //
private:
  std::vector<Material> materials_;

  //
  //
  //
  //
  //
  //
  //
  //
  //
  const Material *ARIA_PROP_GETTER(material)() const {
    if (materials().empty())
      return nullptr;

    return materials().data();
  }

  Material *ARIA_PROP_GETTER(material)() {
    if (materials().empty())
      return nullptr;

    return materials().data();
  }
};

} // namespace ARIA
