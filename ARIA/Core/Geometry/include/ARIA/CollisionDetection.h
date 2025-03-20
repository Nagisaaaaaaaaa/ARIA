#pragma once

#include "ARIA/AABB.h"
#include "ARIA/Triangle.h"

namespace ARIA {

template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool SATImpl(const AABB3<T> &aabb, const Triangle3<T> &tri) {
  using Vec3T = Vec3<T>;

  Vec3T c = aabb.center();
  Vec3T e = aabb.diagonal() / 2_R;

  Triangle v = tri;
  ForEach<3>([&]<auto i>() { v[i] -= c; });

  std::array<Vec3T, 3> f;
  ForEach<3>([&]<auto i>() { f[i] = v[(i + 1) % 3] - v[i]; });

  constexpr C<T(0)> _0;
  constexpr C<T(1)> _1;

  constexpr Tec u0{_1, _0, _0};
  constexpr Tec u1{_0, _1, _0};
  constexpr Tec u2{_0, _0, _1};
  constexpr Tup u{u0, u1, u2};

  auto isSA = [&](const auto &axis) {
    std::array p{Dot(ToTec(v[0]), axis), //
                 Dot(ToTec(v[1]), axis), //
                 Dot(ToTec(v[2]), axis)};

    auto abs = [](const auto &v) { return v < _0 ? -v : v; };
    T r = Dot(ToTec(e), Tec{abs(Dot(u0, axis)), //
                            abs(Dot(u1, axis)), //
                            abs(Dot(u2, axis))});

    return std::max({p[0], p[1], p[2]}) < -r || std::min({p[0], p[1], p[2]}) > r;
  };

  bool isS = false;
  ForEach<3>([&]<auto iU>() {
    ForEach<3>([&]<auto iF>() {
      if (isS)
        return;
      Tec a = Cross(get<iU>(u), ToTec(f[iF]));
      isS |= isSA(a);
    });
  });
  if (isS)
    return false;

  if (isSA(u0) || isSA(u1) || isSA(u2))
    return false;

  if (isSA(tri.stableNormal()))
    return false;

  return true;
}

} // namespace ARIA
