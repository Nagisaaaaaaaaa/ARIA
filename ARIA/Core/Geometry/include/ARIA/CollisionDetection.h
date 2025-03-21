#pragma once

#include "ARIA/detail/CollisionDetectionImpl.h"

namespace ARIA {

/// \brief Test whether the given `aabb` and `tri` are intersecting.
///
/// \example ```cpp
/// AABB3r aabb{...};
/// Triangle3r tri{...};
/// bool collides = DetectCollision(aabb, tri);
/// ```
template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool DetectCollision(const AABB3<T> &aabb, const Triangle3<T> &tri) {
  return collision_detection::detail::SAT(aabb, tri);
}

/// \brief Test whether the given `tri` and `aabb` are intersecting.
template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool DetectCollision(const Triangle3<T> &tri, const AABB3<T> &aabb) {
  return DetectCollision(aabb, tri);
}

} // namespace ARIA
