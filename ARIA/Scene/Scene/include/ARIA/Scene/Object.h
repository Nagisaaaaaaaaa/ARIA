#pragma once

/// \file
/// \brief `Object` is the base class for all entities in ARIA Scenes.
///
/// `Object` is implemented similar to Unity `GameObject`,
/// see https://docs.unity3d.com/ScriptReference/GameObject.html.

//
//
//
//
//
#include "ARIA/Property.h"
#include "ARIA/Registry.h"

#include <boost/iterator/indirect_iterator.hpp>

#include <memory>

namespace ARIA {

// Forward declaration.
class Component;
class Transform;

//
//
//
//
//
/// \brief Base class for all entities in ARIA Scenes.
///
/// `Object` is implemented similar to Unity `GameObject`,
/// see https://docs.unity3d.com/ScriptReference/GameObject.html.
class Object final : public Registry<Object> {
public:
  /// \brief Create a root object. Reference to this object will be returned.
  ///
  /// Please read the comments of `parent` before continue.
  ///
  /// \example ```
  /// Object& obj = Object::Create();
  /// ```
  ///
  /// \warning Life cycles of ARIA `Object`s are designed similar to Unity.
  /// We create `Object`s with `Create()` and destroy them with `Destroy()` or `DestroyImmediate()`.
  /// This design works well in game engines because you usually want to postpone the destruction.
  /// See https://docs.unity3d.com/ScriptReference/Object.Destroy.html.
  ///
  /// `DestroyImmediate()` is provided in this module, while `Destroy()` is not, because
  /// implementation of `Destroy()` depends on the specific lifecycle.
  /// For example, there's one stage in the Unity loop which account for the destruction of
  /// all `GameObject`s which were destroyed with `Destroy()` in the previous update stages.
  ///
  /// So, developers of lifecycles should implement their own `Destroy()`,
  /// for example, postpone the destruction to some time and call `DestroyImmediate()` at that time.
  [[nodiscard]] static Object &Create();

  //
  //
  //
  /// \brief Destroys the object immediately.
  ///
  /// Please read the comments of `Object::Create()` before continue.
  ///
  /// \warning You should never iterate through arrays and destroy the elements you are iterating over.
  /// This will cause serious problems (as a general programming practice, not just in ARIA and Unity).
  ///
  /// So, this function should be carefully handled by the lifecycle.
  ///
  /// \see Destroy
  friend void DestroyImmediate(Object &object);

  /// \brief Destroys the component immediately.
  ///
  /// Please read the comments of `Object::Create()` before continue.
  ///
  /// \warning You should never iterate through arrays and destroy the elements you are iterating over.
  /// This will cause serious problems (as a general programming practice, not just in ARIA and Unity).
  ///
  /// So, this function should be carefully handled by the lifecycle.
  ///
  /// \see Destroy
  friend void DestroyImmediate(Component &component);

  //
  //
  //
public:
  /// \brief Name of the object.
  ///
  /// \example ```cpp
  /// std::string& name = obj.name();
  /// obj.name() = ...;
  /// ```
  ARIA_REF_PROP(public, , name, name_);

  //
  //
  //
  /// \brief The parent object of the current object.
  ///
  /// \example ```cpp
  /// Object* parent = obj.parent();
  ///
  /// Object* newParent = ...;
  /// obj.parent() = newParent; // This will change the parent.
  /// parent = newParent;       //! WARNING, this will not work, see `Auto.h` for the details.
  /// ```
  ///
  /// \warning If `b` is a child of `a`, and one calls `a.parent() = b`,
  /// such cycles will be automatically detected and exceptions will be thrown.
  ///
  /// \warning If the current object is a "root" object, that is, `IsRoot()` returns true,
  /// this function will return reference to the "halo root" object.
  /// The "halo object" is the parent object of all "root" objects.
  ///
  /// The halo root object is introduced to make the hierarchy like a "tree".
  /// That is, the halo root is the actual tree root of the hierarchy tree.
  ///
  /// So, users should not modify anything about the halo root.
  /// Or there will be undefined behaviors.
  ARIA_PROP_BEGIN(public, public, , Object *, parent);
  /**/ ARIA_SUB_PROP_BEGIN(, Object *, parent);
  /**/ /**/ ARIA_SUB_PROP(, Object *, parent);
  /**/ /**/ ARIA_SUB_PROP(, Object *, root);
  /**/ /**/ ARIA_SUB_PROP(, Transform &, transform);
  /**/ ARIA_SUB_PROP_END;
  /**/ ARIA_SUB_PROP_BEGIN(, Object *, root);
  /**/ /**/ ARIA_SUB_PROP(, Object *, parent);
  /**/ /**/ ARIA_SUB_PROP(, Object *, root);
  /**/ /**/ ARIA_SUB_PROP(, Transform &, transform);
  /**/ ARIA_SUB_PROP_END;
  /**/ ARIA_SUB_PROP(, Transform &, transform);
  ARIA_PROP_END;

  /// \brief The root object of the current object.
  ///
  /// Please read the comments of `parent` before continue.
  ///
  /// \example ```cpp
  /// Object* root = obj.root();
  ///
  /// Object* newRoot = ...;
  /// obj.root() = newRoot; // This will set parent of the original root object to `newRoot`.
  ///
  /// \warning Similar to `parent`, cycles will be automatically detected.
  /// ```
  ARIA_PROP_BEGIN(public, public, , Object *, root);
  /**/ ARIA_SUB_PROP_BEGIN(, Object *, parent);
  /**/ /**/ ARIA_SUB_PROP(, Object *, parent);
  /**/ /**/ ARIA_SUB_PROP(, Object *, root);
  /**/ /**/ ARIA_SUB_PROP(, Transform &, transform);
  /**/ ARIA_SUB_PROP_END;
  /**/ ARIA_SUB_PROP_BEGIN(, Object *, root);
  /**/ /**/ ARIA_SUB_PROP(, Object *, parent);
  /**/ /**/ ARIA_SUB_PROP(, Object *, root);
  /**/ /**/ ARIA_SUB_PROP(, Transform &, transform);
  /**/ ARIA_SUB_PROP_END;
  /**/ ARIA_SUB_PROP(, Transform &, transform);
  ARIA_PROP_END;

  //
  //
  //
  /// \brief Get the transform component of the current object.
  /// This property will always return a valid reference to a valid transform because
  /// any object should have and exactly have one transform.
  ///
  /// \example ```cpp
  /// Transform& trans = obj.transform();
  /// ```
  ARIA_REF_PROP(public, , transform, ARIA_PROP_IMPL(transform)());

  //
  //
  //
public:
  /// \brief Whether the current object is a root object.
  ///
  /// Please read the comments of `parent` before continue.
  ///
  /// \example ```cpp
  /// bool isRoot = obj.IsRoot();
  /// ```
  [[nodiscard]] bool IsRoot() const;

  /// \brief Is this object a child (or a grandchild, or .etc) of `parent`?
  ///
  /// \example ```cpp
  /// bool isChildOf = obj.IsChildOf(anotherObj);
  /// ```
  [[nodiscard]] bool IsChildOf(const Object &parent) const;

  //
  //
  //
public:
  /// \brief Add a component with the given type `TComponent` to this object.
  /// Constructor of the added component is called with arguments `ts...`.
  ///
  /// \example ```cpp
  /// Camera& = obj.AddComponent<Camera>(...);
  /// ```
  template <typename TComponent, typename... Ts>
    requires(std::derived_from<TComponent, Component>)
  inline TComponent &AddComponent(Ts &&...ts);

  /// \brief Try to get pointer to a component of type `TComponent` on the specified `Object`.
  /// Returns `nullptr` if components with type `TComponent` not exist.
  ///
  /// \example ```cpp
  /// Camera* obj.GetComponent<Camera>();
  /// ```
  ///
  /// \warning If there are multiple components with the same type, only the first one will be returned.
  template <typename TComponent>
  [[nodiscard]] inline TComponent *GetComponent();

  //
  //
  //
private:
  using Base = Registry<Object>;

public:
  ARIA_COPY_MOVE_ABILITY(Object, delete, delete);

  /// \brief See `Create()` and `DestroyImmediate()`.
  ~Object() = default;

  //
  //
  //
public:
  /// \brief Get the non-const begin iterator of the children `Object`s.
  [[nodiscard]] auto begin() { return boost::make_indirect_iterator(children_.begin()); }

  /// \brief Get the non-const end iterator of the children `Object`s.
  [[nodiscard]] auto end() { return boost::make_indirect_iterator(children_.end()); }

  /// \brief Get the const begin iterator of the children `Object`s.
  [[nodiscard]] auto cbegin() const { return boost::make_indirect_iterator(children_.cbegin()); }

  /// \brief Get the const end iterator of the children `Object`s.
  [[nodiscard]] auto cend() const { return boost::make_indirect_iterator(children_.cend()); }

  //
  //
  //
public:
  /// \brief Get the number of existing `Object`s.
  ///
  /// \example ```cpp
  /// std::cout << Object::size() << std::endl;
  /// ```
  [[nodiscard]] static size_t size() noexcept;

  /// \brief Get the non-const begin iterator of the existing `Object`s.
  ///
  /// \example ```cpp
  /// for (auto it = Object::Begin(); it != Object::End(); ++it) { ... }
  /// ```
  [[nodiscard]] static decltype(Base::begin()) Begin() noexcept;

  /// \brief Get the non-const end iterator of the existing `Object`s.
  [[nodiscard]] static decltype(Base::end()) End() noexcept;

  /// \brief Get the const begin iterator of the existing `Object`s.
  ///
  /// \example ```cpp
  /// for (auto it = Object::CBegin(); it != Object::CEnd(); ++it) { ... }
  /// ```
  [[nodiscard]] static decltype(Base::cbegin()) CBegin() noexcept;

  /// \brief Get the const end iterator of the existing `Object`s.
  [[nodiscard]] static decltype(Base::cend()) CEnd() noexcept;

  /// \brief Get the non-const range of the existing `Object`s.
  ///
  /// \example ```cpp
  /// for (auto &a : Object::Range()) { ... }
  /// ```
  [[nodiscard]] static decltype(Base::range()) Range() noexcept;

  /// \brief Get the const range of the existing `Object`s.
  ///
  /// \example ```cpp
  /// for (auto &a : Object::CRange()) { ... }
  /// ```
  [[nodiscard]] static decltype(Base::crange()) CRange() noexcept;

  //
  //
  //
public:
  /// \brief Two `Object`s are defined as equal when they have exactly the same address.
  bool operator==(const Object &other) const noexcept { return this == &other; }

  /// \brief Two `Object`s are defined as equal when they have exactly the same address.
  bool operator!=(const Object &other) const noexcept { return !operator==(other); }

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
  static std::unique_ptr<Object> haloRoot_;

  std::string name_;
  Object *parent_{};
  std::vector<std::unique_ptr<Object>> children_;
  std::vector<std::unique_ptr<Component>> components_;

  explicit Object(Object *parent);

  //
  //
  //
  [[nodiscard]] const Object *ARIA_PROP_IMPL(parent)() const;
  [[nodiscard]] Object *ARIA_PROP_IMPL(parent)();
  void ARIA_PROP_IMPL(parent)(Object *value);

  [[nodiscard]] const Object *ARIA_PROP_IMPL(root)() const;
  [[nodiscard]] Object *ARIA_PROP_IMPL(root)();
  void ARIA_PROP_IMPL(root)(Object *value);

  [[nodiscard]] const Transform &ARIA_PROP_IMPL(transform)() const;
  [[nodiscard]] Transform &ARIA_PROP_IMPL(transform)();

  //
  //
  //
  // Add a component without checking whether the type is `Transform`.
  template <typename TComponent, typename... Ts>
    requires(std::derived_from<TComponent, Component>)
  TComponent &AddComponentNoTransformCheck(Ts &&...ts);
};

} // namespace ARIA

#include "ARIA/Scene/detail/Object.inc"
