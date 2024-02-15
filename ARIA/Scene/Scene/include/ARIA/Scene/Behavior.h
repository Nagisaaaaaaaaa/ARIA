#pragma once

/// \file
/// \brief `Behaviour`s are `Component`s that can be enabled or disabled.
///
/// `Behavior` is implemented similar to Unity `Behavior`,
/// see https://docs.unity3d.com/ScriptReference/Behaviour.html.

//
//
//
//
//
#include "ARIA/Scene/Component.h"

namespace ARIA {

/// \brief `Behaviour`s are `Component`s that can be enabled or disabled.
///
/// `Behavior` is implemented similar to Unity `Behavior`,
/// see https://docs.unity3d.com/ScriptReference/Behaviour.html.
class Behavior : public Component {
public:
  /// \brief Enabled `Behaviour`s are updated, disabled `Behaviour`s are not.
  ///
  /// \example ```cpp
  /// bool& enabled = b.enabled();
  /// ```
  ARIA_REF_PROP(public, , enabled, enabled_);

  //
  //
  //
protected:
  using Base = Component;

  /// \brief Please read the comments of `Component::Component` before continue.
  using Base::Base;

public:
  ARIA_COPY_MOVE_ABILITY(Behavior, delete, delete);

  /// \brief Please read the comments of `Component::~Component` before continue.
  ~Behavior() override = default;

  //
  //
  //
  //
  //
private:
  bool enabled_{true};
};

} // namespace ARIA
