#pragma once

/// \file
/// \brief `Component` is the base class for everything attached to an `Object`.
///
/// `Component` is implemented similar to Unity `Component`,
/// see https://docs.unity3d.com/ScriptReference/GameObject.html.

//
//
//
//
//
#include "ARIA/Property.h"

namespace ARIA {

// Forward declaration.
class Object;
class Transform;

//
//
//
//
//
/// \brief Base class for everything attached to an `Object`.
///
/// `Component` is implemented similar to Unity `Component`,
/// see https://docs.unity3d.com/ScriptReference/Component.html.
class Component {
public:
  /// \brief The object which this component belongs to.
  ///
  /// \example ```cpp
  /// Object& obj = comp.object();
  /// ```
  ARIA_REF_PROP(public, , object, object_);

  /// \brief Transform of the object, which this component belongs to.
  ///
  /// \example ```cpp
  /// Transform& trans = comp.transform();
  /// ```
  ARIA_REF_PROP(public, , transform, ARIA_PROP_GETTER(transform)());

  //
  //
  //
protected:
  /// \brief Constructors of any `Component`, such as `Transform` and `Camera`,
  /// are only called by `Object::AddComponent()`.
  /// In order to prevent users from accidentally calling it,
  /// constructors should be declared as `protected` or `private`.
  /// Use `friend Object` to give `Object` the right to call them.
  /// See the implementation of `Transform` for how to implement a safe component.
  explicit Component(Object &object) : object_(object) {}

public:
  ARIA_COPY_MOVE_ABILITY(Component, delete, delete);

  /// \brief Use `DestroyImmediate` to destroy a component from an object.
  /// See `Object.h` for more details.
  virtual ~Component() = default;

  //
  //
  //
  //
  //
private:
  Object &object_;

  const Transform &ARIA_PROP_GETTER(transform)() const;
  Transform &ARIA_PROP_GETTER(transform)();
};

} // namespace ARIA
