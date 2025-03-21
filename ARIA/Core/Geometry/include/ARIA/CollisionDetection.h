#pragma once

#include "ARIA/detail/CollisionDetectionImpl.h"

namespace ARIA {

template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool DetectCollision(const AABB3<T> &aabb, const Triangle3<T> &tri) {
  return collision_detection::detail::SAT(aabb, tri);
}

template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool DetectCollision(const Triangle3<T> &tri, const AABB3<T> &aabb) {
  return DetectCollision(aabb, tri);
}

} // namespace ARIA
