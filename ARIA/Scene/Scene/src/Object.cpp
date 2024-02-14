#include "ARIA/Scene/Object.h"
#include "ARIA/Scene/Components/Transform.h"

namespace ARIA {

Object &Object::Create() {
  return *haloRoot_->children_.emplace_back(new Object{haloRoot_.get()});
}

//
//
//
void DestroyImmediate(Object &object) {
  if (object.parent_) [[likely]] {
    auto &siblings = object.parent_->children_;
    auto it = std::find_if(siblings.begin(), siblings.end(),
                           [&](const std::unique_ptr<Object> &child) { return child.get() == &object; });

    siblings.erase(it);
  } else {
    throw std::runtime_error("Should not destroy the halo root object");
  }
}

void DestroyImmediate(Component &component) {
  if (dynamic_cast<Transform *>(&component)) {
    throw std::runtime_error("Should not destroy the `Transform` component");
  }

  if (component.object().parent_) [[likely]] {
    auto &components = component.object().components_;
    auto it = std::find_if(components.begin(), components.end(),
                           [&](const std::unique_ptr<Component> &comp) { return comp.get() == &component; });

    components.erase(it);
  } else {
    throw std::runtime_error("Should not destroy component of the halo root object");
  }
}

//
//
//
const Object *Object::ARIA_PROP_IMPL(parent)() const {
  return parent_;
}

Object *Object::ARIA_PROP_IMPL(parent)() {
  return parent_;
}

void Object::ARIA_PROP_IMPL(parent)(Object *value) {
  if (value->IsChildOf(*this)) {
    throw std::exception{};
  }

  if (parent_ == value) [[unlikely]] {
    return;
  }

  std::unique_ptr<Object> self;
  if (parent_) [[likely]] {
    auto &siblings = parent_->children_;
    auto it = std::find_if(siblings.begin(), siblings.end(),
                           [this](const std::unique_ptr<Object> &child) { return child.get() == this; });

    self = std::move(*it);
    siblings.erase(it);
  }

  parent_ = value;
  value->children_.emplace_back(std::move(self));
}

const Object *Object::ARIA_PROP_IMPL(root)() const {
  const Object *p = this;

  while (!p->IsRoot())
    p = p->parent();

  return p;
}

Object *Object::ARIA_PROP_IMPL(root)() {
  Object *p = this;

  while (!p->IsRoot())
    p = p->parent();

  return p;
}

void Object::ARIA_PROP_IMPL(root)(Object *value) {
  if (value->IsChildOf(*this)) {
    throw std::exception{};
  }

  //! Should not write `root()->parent() = value`,
  //! or the setter of `root` will be called during this expression,
  //! which will result in an infinite loop.
  Object *r = root();
  r->parent() = value;
}

//
//
//
const Transform &Object::ARIA_PROP_IMPL(transform)() const {
  return *static_cast<Transform *>(components_[0].get());
}

Transform &Object::ARIA_PROP_IMPL(transform)() {
  return *static_cast<Transform *>(components_[0].get());
}

//
//
//
bool Object::IsRoot() const {
  return parent() == haloRoot_.get();
}

bool Object::IsChildOf(const Object &parent) const {
  const Object *p = this;

  while (p->parent()) {
    p = p->parent();

    if (p == &parent)
      return true;
  }

  return false;
}

//
//
//
std::unique_ptr<Object> Object::haloRoot_ = std::unique_ptr<Object>(new Object{nullptr});

Object::Object(Object *parent) : parent_(parent) {
  AddComponentNoTransformCheck<Transform>();
}

} // namespace ARIA
