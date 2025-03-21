#include "ARIA/CollisionDetection.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

template <typename T, uint d>
ARIA_HOST_DEVICE bool IsIn(const Vec<T, d> &p, const AABB<T, d> &aabb) {
  bool res = true;
  ForEach<d>([&]<auto i>() { res = res && p[i] >= aabb.inf()[i] && p[i] <= aabb.sup()[i]; });
  return res;
}

} // namespace

TEST(CollisionDetection, Base) {}

} // namespace ARIA
