#include "ARIA/Scene/Components/Transform.h"
#include "ARIA/Scene/Object.h"

namespace ARIA {

const Transform &Component::ARIA_PROP_IMPL(transform)() const {
  return object().transform();
}

Transform &Component::ARIA_PROP_IMPL(transform)() {
  return object().transform();
}

} // namespace ARIA
