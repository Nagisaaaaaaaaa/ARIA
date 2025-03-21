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

template <typename T, uint d>
ARIA_HOST_DEVICE T DistSquared(const Vec<T, d> &p, const AABB<T, d> &aabb) {
  T distSq(0);
  ForEach<d>([&]<auto i>() {
    if (p[i] < aabb.inf()[i])
      distSq += Pow<2>(aabb.inf()[i] - p[i]);
    if (p[i] > aabb.sup()[i])
      distSq += Pow<2>(p[i] - aabb.sup()[i]);
  });
  return distSq;
}

template <typename T, typename F>
void ForEachDivision(uint division, const Triangle3<T> &tri, const F &f) {
  using Vec3T = Vec3<T>;

  if (division == 0) {
    f(tri);
  } else {
    Vec3T p0 = tri[0];
    Vec3T p1 = tri[1];
    Vec3T p2 = tri[2];

    Vec3T p01 = (p0 + p1) / 2;
    Vec3T p12 = (p1 + p2) / 2;
    Vec3T p20 = (p2 + p0) / 2;

    ForEachDivision(division - 1, {p0, p01, p20}, f);
    ForEachDivision(division - 1, {p01, p1, p12}, f);
    ForEachDivision(division - 1, {p01, p12, p20}, f);
    ForEachDivision(division - 1, {p20, p12, p2}, f);
  }
}

template <typename T>
bool Collide(const AABB3<T> &aabb, const Triangle3<T> &tri) {
  T distSq = infinity<T>;
  ForEachDivision(16, tri, [&](const Triangle3<T> &t) {
    distSq = std::min({distSq, DistSquared(t[0], aabb), DistSquared(t[1], aabb), DistSquared(t[2], aabb)});
  });
  return distSq == T(0);
}

} // namespace

TEST(CollisionDetection, Base) {}

} // namespace ARIA
