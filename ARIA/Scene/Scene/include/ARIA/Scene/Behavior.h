#pragma once

#include "Component.h"

namespace ARIA {

/// \brief Behaviours are `Component`s that can be enabled or disabled.
///
/// See https://docs.unity3d.com/ScriptReference/Behaviour.html.
class Behavior : public Component {
public:
  /// \brief Enabled Behaviours are Updated, disabled Behaviours are not.
  ///
  /// \example ```cpp
  /// bool& enabled = b.enabled();
  /// ```
  ARIA_REF_PROP(public, , enabled, enabled_);

protected:
  using Base = Component;
  using Base::Base;

public:
  Behavior(const Behavior &) = delete;
  Behavior(Behavior &&) noexcept = delete;
  Behavior &operator=(const Behavior &) = delete;
  Behavior &operator=(Behavior &&) noexcept = delete;
  ~Behavior() override = default;

private:
  bool enabled_{true};
};

} // namespace ARIA
