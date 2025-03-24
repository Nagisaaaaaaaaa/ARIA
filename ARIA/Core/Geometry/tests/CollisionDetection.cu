#include "ARIA/CollisionDetection.h"
#include "ARIA/Launcher.h"

#include <gtest/gtest.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace ARIA {

namespace {

template <uint division, typename T, typename F>
ARIA_HOST_DEVICE void ForEachDivision(const LineSegment2<T> &seg, const F &f) {
  for (uint i = 0; i < division; ++i) {
    T w = T(i) / T(division - 1);

    Vec2<T> pos = Lerp(seg[0], seg[1], w);
    f(pos);
  }
}

template <uint division, typename T, typename F>
ARIA_HOST_DEVICE void ForEachDivision(const Triangle3<T> &tri, const F &f) {
  for (uint z = 0; z < division; ++z)
    for (uint y = 0; y < division - z; ++y) { // `y + z <= division - 1`.
      uint x = (division - 1) - y - z;        // `x + y + z == division - 1`.

      T wX = T(x) / T(division - 1);
      T wY = T(y) / T(division - 1);
      T wZ = T(z) / T(division - 1);

      Vec3<T> pos = wX * tri[0] + wY * tri[1] + wZ * tri[2];
      f(pos);
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
  ForEachDivision<128>(tri, [&](const Vec3<T> &p) { distSq = std::min(distSq, DistSquared(aabb, p)); });
  return distSq;
}

void Test_AABBTriangle() {
  AABB3f aabb{Vec3f{-31.4159F, 12.34567F, -98.76543F}, Vec3f(95.1413F, 66.66666F, -11.4514F)};
  AABB3f aabbRelaxed = aabb; // A little bit smaller.
  aabbRelaxed.inf() += aabb.diagonal() * 0.0001F;
  aabbRelaxed.sup() -= aabb.diagonal() * 0.0001F;

  Vec3f pMin = aabb.inf() - aabb.diagonal();
  Vec3f pMax = aabb.sup() + aabb.diagonal();

  constexpr uint n = 20;

  thrust::host_vector<Vec3f> psH;
  for (uint z = 0; z < n; ++z)
    for (uint y = 0; y < n; ++y)
      for (uint x = 0; x < n; ++x) {
        Vec3f p = pMin + (pMax - pMin).cwiseProduct(Vec3f(x, y, z) / static_cast<float>(n - 1));
        psH.push_back(p);
      }

  thrust::device_vector<Vec3f> psD = psH;
  int nPs = psD.size();

  Launcher(make_layout_major(nPs, nPs, nPs), [aabb, aabbRelaxed, ps = psD.data()] ARIA_DEVICE(int x, int y, int z) {
    Triangle3f tri{ps[x], ps[y], ps[z]};

    bool collides = DetectCollision(aabb, tri);
    bool collides1 = DetectCollision(tri, aabb);
    ARIA_ASSERT(collides == collides1);

    float distSqRelaxed = DistSquared(aabbRelaxed, tri); // A little bit larger.
    if (collides)
      ARIA_ASSERT(distSqRelaxed < 1.05F);
    else
      ARIA_ASSERT(distSqRelaxed > 0.0F);
  }).Launch();

  cuda::device::current::get().synchronize();
}

} // namespace

TEST(CollisionDetection, D3) {
  Test_AABBTriangle();
}

} // namespace ARIA
