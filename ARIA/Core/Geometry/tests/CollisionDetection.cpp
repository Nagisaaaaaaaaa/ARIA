#include "ARIA/CollisionDetection.h"

#include <gtest/gtest.h>

#include <random>

namespace ARIA {

namespace {

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

    ForEachDivision<T, F>(division - 1, {p0, p01, p20}, f);
    ForEachDivision<T, F>(division - 1, {p01, p1, p12}, f);
    ForEachDivision<T, F>(division - 1, {p01, p12, p20}, f);
    ForEachDivision<T, F>(division - 1, {p20, p12, p2}, f);
  }
}

template <typename T, uint d>
T DistSquared(const AABB<T, d> &aabb, const Vec<T, d> &p) {
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
T DistSquared(const AABB3<T> &aabb, const Triangle3<T> &tri) {
  T distSq = infinity<T>;
  ForEachDivision(6, tri, [&](const Triangle3<T> &t) {
    distSq = std::min({distSq, DistSquared(aabb, t[0]), DistSquared(aabb, t[1]), DistSquared(aabb, t[2])});
  });
  return distSq;
}

} // namespace

TEST(CollisionDetection, Base) {
  std::mt19937 gen(0);
  std::uniform_real_distribution<float> dis(-100.0F, 100.0F);

  AABB3f aabb{Vec3f{dis(gen), dis(gen), dis(gen)}, Vec3f(dis(gen), dis(gen), dis(gen))};

  Vec3f pMin = aabb.inf() - aabb.diagonal();
  Vec3f pMax = aabb.sup() + aabb.diagonal();

  constexpr int n = 5;

  std::vector<Vec3f> ps;
  for (int z = 0; z < n; ++z)
    for (int y = 0; y < n; ++y)
      for (int x = 0; x < n; ++x) {
        Vec3f p = pMin + (pMax - pMin).cwiseProduct(Vec3f(x, y, z) / static_cast<float>(n - 1));
        ps.emplace_back(p);
      }

  for (const auto &p0 : ps)
    for (const auto &p1 : ps)
      for (const auto &p2 : ps) {
        Triangle3f tri{p0, p1, p2};

        bool collide = SATImpl(aabb, tri);
        float distSq = DistSquared(aabb, tri);

        if (collide)
          EXPECT_LT(distSq, 1.0F);
        else
          EXPECT_GT(distSq, 1e-5F);
      }
}

} // namespace ARIA
