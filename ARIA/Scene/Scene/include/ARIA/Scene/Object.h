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

  Object(const Object &) = delete;
  Object(Object &&) noexcept = delete;
  Object &operator=(const Object &) = delete;
  Object &operator=(Object &&) noexcept = delete;
  ~Object() = default;

  //
  //
  //
  //
  //
private:
  // Iterators and ranges fwd.
  template <typename TObjectMaybeConst>
  class JoinIterator;
  class JoinComponentIterator;

  template <typename TItBegin, typename TItEnd>
  class Range;

public:
  /// \brief Get the joined object range.
  /// For example, the object hierarchy looks like:
  /// ```
  /// o0's children: [o1, o2]
  /// o1's children: [o3, o4]
  /// o2's children: [o5]
  /// o3's children: [o6]
  /// ```
  /// Then, the traversal order is: `0, 1, 3, 6, 4, 2, 5`.
  ///
  /// \example ```cpp
  /// for (auto &obj : Object::rangeJoined()) {
  ///   std::cout << obj.name() << std::endl;
  /// }
  /// ```
  static Range<JoinIterator<Object>, JoinIterator<Object>> rangeJoined();

  /// \brief Get the joined object range.
  /// For example, the object hierarchy looks like:
  /// ```
  /// o0's children: [o1, o2]
  /// o1's children: [o3, o4]
  /// o2's children: [o5]
  /// o3's children: [o6]
  /// ```
  /// Then, the traversal order is: `0, 1, 3, 6, 4, 2, 5`.
  ///
  /// \example ```cpp
  /// for (auto &obj : Object::crangeJoined()) {
  ///   std::cout << obj.name() << std::endl;
  /// }
  /// ```
  static Range<JoinIterator<const Object>, JoinIterator<const Object>> crangeJoined();

  /// \brief Get the joined component range.
  /// Similar to `rangeJoined`, but iterate through each component of each object.
  ///
  /// \example ```cpp
  /// for (auto &comp : Object::componentRangeJoined()) {
  ///   std::cout << comp.object().name() << std::endl;
  /// }
  /// ```
  static Range<JoinComponentIterator, JoinComponentIterator> componentRangeJoined();

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
  template <typename TObjectMaybeConst>
  class JoinIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = TObjectMaybeConst;
    using pointer = TObjectMaybeConst *;
    using reference = TObjectMaybeConst &;

    //! The ranges library requires that iterators should be default constructable,
    //! so iterators are constructed to `end` by default.
    JoinIterator() noexcept = default;

    explicit JoinIterator(pointer ptr) {
      if (ptr) {
        stack_.push(ptr);

        ++*this;
      }
    }

    reference operator*() const { return *stack_.top(); }

    pointer operator->() noexcept { return stack_.top(); }

    // ++it
    JoinIterator &operator++() {
      if (!stack_.empty()) {
        pointer current = stack_.top();
        stack_.pop();

        for (auto it = current->children_.rbegin(); it != current->children_.rend(); ++it) {
          stack_.push(it->get());
        }
      }
      return *this;
    }

    // it++
    JoinIterator operator++(int) {
      JoinIterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const JoinIterator &a, const JoinIterator &b) noexcept {
      return (a.stack_.empty() && b.stack_.empty()) ||
             (!a.stack_.empty() && !b.stack_.empty() && a.stack_.top() == b.stack_.top());
    }

    friend bool operator!=(const JoinIterator &a, const JoinIterator &b) noexcept { return !(a == b); }

    JoinIterator(const JoinIterator &) noexcept = default;
    JoinIterator(JoinIterator &&) noexcept = default;
    JoinIterator &operator=(const JoinIterator &) noexcept = default;
    JoinIterator &operator=(JoinIterator &&) noexcept = default;

  private:
    std::stack<pointer> stack_{};
  };

  static_assert(std::forward_iterator<JoinIterator<Object>> && std::forward_iterator<JoinIterator<const Object>>,
                "The object join iterators should be at least forward to cooperate with the ranges library");

  //
  //
  //
  class JoinComponentIterator {
  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = Component;
    using pointer = Component *;
    using reference = Component &;

    JoinComponentIterator() noexcept = default;

    explicit JoinComponentIterator(Object *objectPtr) {
      if (objectPtr) {
        stack_.push(objectPtr);
        itComp_ = stack_.top()->components_.begin();

        ++*this;
      }
    }

    reference operator*() const { return **itComp_; }

    pointer operator->() noexcept { return itComp_->get(); }

    // ++it
    JoinComponentIterator &operator++() {
      if (!stack_.empty()) {
        ++itComp_;

        if (itComp_ == stack_.top()->components_.end()) {
          Object *current = stack_.top();
          stack_.pop();

          for (auto it = current->children_.rbegin(); it != current->children_.rend(); ++it) {
            stack_.push(it->get());
          }

          if (!stack_.empty())
            itComp_ = stack_.top()->components_.begin();
          else
            itComp_ = {};
        }
      }
      return *this;
    }

    // it++
    JoinComponentIterator operator++(int) {
      JoinComponentIterator tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const JoinComponentIterator &a, const JoinComponentIterator &b) noexcept {
      return ((a.stack_.empty() && b.stack_.empty()) ||
              (!a.stack_.empty() && !b.stack_.empty() && a.stack_.top() == b.stack_.top())) &&
             (a.itComp_ == b.itComp_);
    }

    friend bool operator!=(const JoinComponentIterator &a, const JoinComponentIterator &b) noexcept {
      return !(a == b);
    }

    JoinComponentIterator(const JoinComponentIterator &) noexcept = default;
    JoinComponentIterator(JoinComponentIterator &&) noexcept = default;
    JoinComponentIterator &operator=(const JoinComponentIterator &) noexcept = default;
    JoinComponentIterator &operator=(JoinComponentIterator &&) noexcept = default;

  private:
    std::stack<Object *> stack_{};
    std::vector<std::unique_ptr<Component>>::iterator itComp_{};
  };

  static_assert(std::forward_iterator<JoinComponentIterator>,
                "The component join iterators should be at least forward to cooperate with the ranges library");

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
