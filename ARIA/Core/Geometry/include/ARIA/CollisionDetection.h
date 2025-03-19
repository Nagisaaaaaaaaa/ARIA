#pragma once

#include "ARIA/AABB.h"
#include "ARIA/Triangle.h"

namespace ARIA {

template <typename T, uint d>
[[nodiscard]] ARIA_HOST_DEVICE static bool SATImpl(const AABB<T, d> &aabb, const Triangle<T, d> &tri) {
}

} // namespace ARIA
