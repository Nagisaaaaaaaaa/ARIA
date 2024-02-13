#pragma once

#include "ARIA/Vec.h"
#include "Shader.h"

namespace ARIA {

/// \brief The material class.
///
/// This class exposes all properties from a material, allowing you to animate them.
/// You can also use it to set custom shader properties that
/// can't be accessed through the inspector (e.g. matrices).
///
/// In order to get the material used by an object, use the `Renderer.material` property.
///
/// See https://docs.unity3d.com/ScriptReference/Material.html.
class Material {
  /// \brief The main color of the material.
  ARIA_REF_PROP(public, , color, color_)

  /// \brief The shader used by the material.
  ARIA_REF_PROP(public, , shader, shader_)

private:
  Vec4r color_;
  Shader *shader_;
};

} // namespace ARIA
