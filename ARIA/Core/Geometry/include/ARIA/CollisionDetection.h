#pragma once

/// \file
/// \brief Utilities for testing whether the given primitives are intersecting.

//
//
//
//
//
#include "ARIA/detail/CollisionDetectionImpl.h"

namespace ARIA {

/// \brief Test whether the given `aabb` and `seg` are intersecting.
///
/// \example ```cpp
/// AABB2r aabb{...};
/// LineSegment2r seg{...};
/// bool collides = DetectCollision(aabb, seg);
/// ```
template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool DetectCollision(const AABB2<T> &aabb, const LineSegment2<T> &seg) {
  return collision_detection::detail::SAT(aabb, seg);
}

/// \brief Test whether the given `seg` and `aabb` are intersecting.
///
/// \example ```cpp
/// LineSegment2r seg{...};
/// AABB2r aabb{...};
/// bool collides = DetectCollision(seg, aabb);
/// ```
template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool DetectCollision(const LineSegment2<T> &seg, const AABB2<T> &aabb) {
  return DetectCollision(aabb, seg);
}

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
///
/// \example ```cpp
/// Triangle3r tri{...};
/// AABB3r aabb{...};
/// bool collides = DetectCollision(tri, aabb);
/// ```
template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool DetectCollision(const Triangle3<T> &tri, const AABB3<T> &aabb) {
  return DetectCollision(aabb, tri);
}

} // namespace ARIA
