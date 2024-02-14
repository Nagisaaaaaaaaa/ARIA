#pragma once

#include "ARIA/Property.h"

#include <memory>
#include <stack>

namespace ARIA {

// Fwd.
class Component;
class Transform;

//
//
//
/// \brief Base class for all entities in ARIA Scenes.
///
/// See https://docs.unity3d.com/ScriptReference/GameObject.html.
class Object final {
public:
  /// \brief Name of the object.
  ///
  /// \example ```cpp
  /// std::string& name = o.name();
  /// ```
  ARIA_REF_PROP(public, , name, name_);

  /// \brief The parent object of the current object.
  ///
  /// \example ```cpp
  /// Object& parent = o.parent();
  /// ```
  ///
  /// \note If the current object is a "root" object, that is, `IsRoot()` returns true,
  /// this function will return reference to the "halo root" object.
  /// The "halo object" is the parent object of all "root" objects.
  ///
  /// The halo root object is introduced to make the hierarchy like a "tree".
  /// That is, the halo root is the actual tree root of the hierarchy tree.
  ///
  /// So, users should not modify anything about the halo root.
  /// Or there will be undefined behaviors.
  ARIA_PROP_BEGIN(public, public, , Object *, parent);
  ARIA_SUB_PROP(, Object *, parent);
  ARIA_SUB_PROP(, Object *, root);
  ARIA_SUB_PROP(, Transform &, transform);
  ARIA_PROP_END;

  /// \brief Get the "root" object of the current object.
  /// See `parent` for more details.
  ///
  /// \example ```cpp
  /// Object& root = o.root();
  /// ```
  ARIA_PROP_BEGIN(public, public, , Object *, root);
  ARIA_SUB_PROP(, Object *, parent);
  ARIA_SUB_PROP(, Object *, root);
  ARIA_SUB_PROP(, Transform &, transform);
  ARIA_PROP_END;

  /// \brief Get the transform component of the current object.
  /// This property will always return a valid reference to a valid transform because
  /// any object should have and exactly have one transform.
  ///
  /// \example ```cpp
  /// Transform& t = o.transform();
  /// ```
  ARIA_REF_PROP(public, , transform, ARIA_PROP_IMPL(transform)());

public:
  /// \brief Whether the current object is a "root" object.
  /// See `parent` for more details.
  ///
  /// \example ```cpp
  /// bool isRoot = t.IsRoot();
  /// ```
  [[nodiscard]] bool IsRoot() const;

  /// \brief Is this object a child of `parent`?
  [[nodiscard]] bool IsChildOf(const Object &parent) const;

public:
  /// \brief Add a component with the given type `TComponent` to this object.
  /// Constructor of the added component is called with arguments `ts...`.
  ///
  /// \example ```cpp
  /// Camera& = o.AddComponent<Camera>(...);
  /// ```
  template <typename TComponent, typename... Ts>
    requires(std::derived_from<TComponent, Component>)
  TComponent &AddComponent(Ts &&...ts) {
    static_assert(!std::is_same_v<TComponent, Transform>, "Any object should have and exactly have on `Transform`");

    return AddComponentNoCheck<TComponent>(std::forward<Ts>(ts)...);
  }

  /// \brief Gets a reference to a component of type `TComponent` on the specified `Object`.
  ///
  /// \example ```cpp
  /// Camera* o.GetComponent<Camera>();
  /// ```
  template <typename TComponent>
  TComponent *GetComponent() {
    static_assert(!std::is_same_v<TComponent, Transform>, "Directly call `transform()` instead, which is faster");

    TComponent *t;
    for (const auto &c : components_) {
      if ((t = dynamic_cast<TComponent *>(c.get())))
        break;
    }

    return t;
  }

public:
  /// \brief Create a "root" object, see `parent` for more details.
  static Object &Create();

  ARIA_COPY_MOVE_ABILITY(Object, delete, delete);
  ~Object() = default;

  //
  //
  //
  //
  //
private:
  // Fwd.
  template <typename TItBegin, typename TItEnd>
  class Range;

  //
  //
  //
  //
  //
private:
  std::string name_;
  Object *parent_{};
  std::vector<std::unique_ptr<Object>> children_;
  std::vector<std::unique_ptr<Component>> components_;

  //
  //
  //
  //
  //
  //
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
  //
  //
private:
  static std::unique_ptr<Object> haloRoot_;

  explicit Object(Object *parent);

  //
  //
  //
  //
  //
  template <typename TComponent, typename... Ts>
    requires(std::derived_from<TComponent, Component>)
  TComponent &AddComponentNoCheck(Ts &&...ts) {
    TComponent *component = new TComponent(*this, std::forward<Ts>(ts)...);
    components_.emplace_back(component);

    return *component;
  }

  //
  //
  //
  //
  //
  template <typename TItBegin, typename TItEnd>
  class Range {
  public:
    Range(const TItBegin &begin, const TItEnd &end) : begin_(begin), end_(end) {}

    TItBegin begin() const { return begin_; }

    TItEnd end() const { return end_; }

    Range(const Range &) = default;
    Range(Range &&) noexcept = default;
    Range &operator=(const Range &) = default;
    Range &operator=(Range &&) noexcept = default;

  private:
    TItBegin begin_;
    TItEnd end_;
  };

  //
  //
  //
  //
  //
  friend void DestroyImmediate(Object &object);
  friend void DestroyImmediate(Component &component);
};

//
//
//
//
//
/// \brief Destroys the object immediately.
///
/// \warning You should never iterate through arrays and destroy the elements you are iterating over.
/// This will cause serious problems (as a general programming practice, not just in ARIA and Unity).
///
/// So, this function should be carefully handled by the lifecycle.
///
/// \see Destroy
void DestroyImmediate(Object &object);

/// \brief Destroys the component immediately.
///
/// \warning You should never iterate through arrays and destroy the elements you are iterating over.
/// This will cause serious problems (as a general programming practice, not just in ARIA and Unity).
///
/// So, this function should be carefully handled by the lifecycle.
///
/// \see Destroy
void DestroyImmediate(Component &component);

} // namespace ARIA
