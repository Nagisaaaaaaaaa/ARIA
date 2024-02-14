#pragma once

#include "ARIA/Property.h"

namespace ARIA {

// Fwd.
class Object;
class Transform;

//
//
//
/// \brief Base class for everything attached to a `Object`.
///
/// See https://docs.unity3d.com/ScriptReference/Component.html.
class Component {
public:
  /// \brief The object which this component belongs to.
  ///
  /// \example ```cpp
  /// Object& o = c.object();
  /// ```
  ARIA_REF_PROP(public, , object, object_);

  /// \brief Transform of the object, which this component belongs to.
  ///
  /// \example ```cpp
  /// Transform& t = c.transform();
  /// ```
  ARIA_REF_PROP(public, , transform, ARIA_PROP_IMPL(transform)());

protected:
  explicit Component(Object &object) : object_(object) {}

public:
  ARIA_COPY_MOVE_ABILITY(Component, delete, delete);
  virtual ~Component() = default;

private:
  Object &object_;

  const Transform &ARIA_PROP_IMPL(transform)() const;
  Transform &ARIA_PROP_IMPL(transform)();
};

} // namespace ARIA
