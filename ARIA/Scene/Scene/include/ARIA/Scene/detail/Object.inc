#pragma once

namespace ARIA {

void DestroyImmediate(Object &object);
void DestroyImmediate(Component &component);

//
//
//
//
//
template <typename TComponent, typename... Ts>
  requires(std::derived_from<TComponent, Component>)
inline TComponent &Object::AddComponent(Ts &&...ts) {
  static_assert(std::is_same_v<TComponent, std::decay_t<TComponent>>, "`TComponent` should be a decayed type");

  static_assert(!std::is_same_v<TComponent, Transform>, "Any object should have and exactly have one `Transform`");

  return AddComponentNoTransformCheck<TComponent>(std::forward<Ts>(ts)...);
}

template <typename TComponent>
inline TComponent *Object::GetComponent() {
  static_assert(std::is_same_v<TComponent, std::decay_t<TComponent>>, "`TComponent` should be a decayed type");

  // Optimized for `Transform`.
  if constexpr (std::is_same_v<TComponent, Transform>) {
    return &transform();
  }

  TComponent *t = nullptr;
  for (const auto &c : components_) {
    if ((t = dynamic_cast<TComponent *>(c.get())))
      break;
  }

  return t;
}

//
//
//
template <typename TComponent, typename... Ts>
  requires(std::derived_from<TComponent, Component>)
TComponent &Object::AddComponentNoTransformCheck(Ts &&...ts) {
  TComponent *component = new TComponent(*this, std::forward<Ts>(ts)...);
  components_.emplace_back(component);

  return *component;
}

} // namespace ARIA
