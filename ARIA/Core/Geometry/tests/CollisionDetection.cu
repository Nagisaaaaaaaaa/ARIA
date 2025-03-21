#include "ARIA/CollisionDetection.h"
#include "ARIA/Launcher.h"
#include "ARIA/Vector.h"

#include <gtest/gtest.h>

namespace ARIA {

namespace {

template <uint division, typename T, typename F>
ARIA_HOST_DEVICE void ForEachDivision(const Triangle3<T> &tri, const F &f) {
  using Vec3T = Vec3<T>;

  if constexpr (division == 0) {
    f(tri);
  } else {
    Vec3T p0 = tri[0];
    Vec3T p1 = tri[1];
    Vec3T p2 = tri[2];

    Vec3T p01 = (p0 + p1) / 2;
    Vec3T p12 = (p1 + p2) / 2;
    Vec3T p20 = (p2 + p0) / 2;

    ForEachDivision<division - 1, T>({p0, p01, p20}, f);
    ForEachDivision<division - 1, T>({p01, p1, p12}, f);
    ForEachDivision<division - 1, T>({p01, p12, p20}, f);
    ForEachDivision<division - 1, T>({p20, p12, p2}, f);
  }
}

template <typename T, uint d>
ARIA_HOST_DEVICE T DistSquared(const AABB<T, d> &aabb, const Vec<T, d> &p) {
  T distSq(0);
  ForEach<d>([&]<auto i>() {
    if (p[i] < aabb.inf()[i])
      distSq += Pow<2>(aabb.inf()[i] - p[i]);
    if (p[i] > aabb.sup()[i])
      distSq += Pow<2>(p[i] - aabb.sup()[i]);
  });
  return distSq;
}

template <typename T>
ARIA_HOST_DEVICE T DistSquared(const AABB3<T> &aabb, const Triangle3<T> &tri) {
  T distSq = infinity<T>;
  ForEachDivision<6>(tri, [&](const Triangle3<T> &t) {
    distSq = std::min({distSq, DistSquared(aabb, t[0]), DistSquared(aabb, t[1]), DistSquared(aabb, t[2])});
  });
  return distSq;
}

void TestSAT_AABBTriangle() {
  AABB3f aabb{Vec3f{-31.4159F, 12.34567F, -98.76543F}, Vec3f(95.1413F, 66.66666F, -11.4514F)};
  AABB3f aabbRelaxed = aabb;
  aabbRelaxed.inf() += aabb.diagonal() * 0.0001F;
  aabbRelaxed.sup() -= aabb.diagonal() * 0.0001F;

  Vec3f pMin = aabb.inf() - aabb.diagonal();
  Vec3f pMax = aabb.sup() + aabb.diagonal();

  constexpr int n = 10;

  VectorHost<Vec3f> psH;
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        Vec3f p = pMin + (pMax - pMin).cwiseProduct(Vec3f(x, y, z) / static_cast<float>(n - 1));
        psH.push_back(p);
      }

  VectorDevice<Vec3f> psD = psH;
  int nPs = psD.size();

  Launcher(make_layout_major(nPs, nPs, nPs), [aabb, aabbRelaxed, ps = psD.data()] ARIA_DEVICE(int x, int y, int z) {
    Triangle3f tri{ps[x], ps[y], ps[z]};

    bool collide = SATImpl(aabb, tri);
    float distSqRelaxed = DistSquared(aabbRelaxed, tri);

    if (collide) {
      ARIA_ASSERT(distSqRelaxed < Pow<2>(2.0F));
    } else {
      ARIA_ASSERT(distSqRelaxed > 0);
    }
  }).Launch();

  cuda::device::current::get().synchronize();
}

} // namespace

TEST(CollisionDetection, Base) {
  TestSAT_AABBTriangle();
}

} // namespace ARIA
