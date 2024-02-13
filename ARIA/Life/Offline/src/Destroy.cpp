#include "ARIA/Life/Offline/Destroy.h"
#include "ARIA/Scene/Component.h"
#include "ARIA/Scene/Object.h"

#include <queue>

namespace ARIA {

static std::queue<Object *> destroyedObjects_;
static std::queue<Component *> destroyedComponents_;

static std::queue<Object *> &destroyedObjects() {
  return destroyedObjects_;
}

static std::queue<Component *> &destroyedComponents() {
  return destroyedComponents_;
}

//
//
//
//
//
void Destroy(Object &object) {
  destroyedObjects().push(&object);
}

void Destroy(Component &component) {
  destroyedComponents().push(&component);
}

//
//
//
//
//
void DestructAllDestroyed() {
  while (!destroyedObjects().empty()) {
    Object *object = destroyedObjects().front();
    destroyedObjects().pop();

    DestroyImmediate(*object);
  }

  while (!destroyedComponents().empty()) {
    Component *component = destroyedComponents().front();
    destroyedComponents().pop();

    DestroyImmediate(*component);
  }
}

} // namespace ARIA
