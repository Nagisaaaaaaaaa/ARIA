#pragma once

#include "ARIA/AABB.h"
#include "ARIA/LineSegment.h"
#include "ARIA/Triangle.h"

namespace ARIA {

namespace collision_detection::detail {

//! The following SAT algorithms are implemented based on
//! 2005, Christer Ericson, Real-Time Collision Detection, Chapter 5.2.9 Testing AABB Against Triangle.

template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool SAT(const AABB2<T> &aabb, const LineSegment2<T> &seg) {
  using Vec2T = Vec2<T>;

  Vec2T c = aabb.center();
  Vec2T e = aabb.diagonal() / 2_R;

  LineSegment v = seg;
  ForEach<2>([&]<auto i>() { v[i] -= c; });

  Vec2T f = v[1] - v[0];

  //! `Tec` is intensively used in the following codes to ensure that
  //! expressions will be simplified at compile-time.
  constexpr C<T(0)> _0;
  constexpr C<T(1)> _1;

  constexpr Tec u0{_1, _0};
  constexpr Tec u1{_0, _1};
  constexpr Tup u{u0, u1};

  // Test whether the given `axis` is a separating axis (SA).
  auto isSA = [&](const auto &axis) {
    std::array p{Dot(ToTec(v[0]), axis), //
                 Dot(ToTec(v[1]), axis)};

    auto abs = [](const auto &v) { return v < 0 ? -v : v; };
    T r = Dot(ToTec(e), Tec{abs(Dot(u0, axis)), //
                            abs(Dot(u1, axis))});

    return std::max(p[0], p[1]) < -r || std::min(p[0], p[1]) > r;
  };

  // Whether the two primitives are separating.
  bool isS = false;
  // Test 2 edge normals from the AABB.
  isS = isS || isSA(u0) || isSA(u1);
  // Test 1 edge normal from the line segment.
  isS = isS || isSA(Tec{-f[1], f[0]});

  return !isS;
}

template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool SAT(const AABB3<T> &aabb, const Triangle3<T> &tri) {
  using Vec3T = Vec3<T>;

  Vec3T c = aabb.center();
  Vec3T e = aabb.diagonal() / 2_R;

  Triangle v = tri;
  ForEach<3>([&]<auto i>() { v[i] -= c; });

  std::array<Vec3T, 3> f;
  ForEach<3>([&]<auto i>() { f[i] = v[(i + 1) % 3] - v[i]; });

  //! `Tec` is intensively used in the following codes to ensure that
  //! expressions will be simplified at compile-time.
  constexpr C<T(0)> _0;
  constexpr C<T(1)> _1;

  constexpr Tec u0{_1, _0, _0};
  constexpr Tec u1{_0, _1, _0};
  constexpr Tec u2{_0, _0, _1};
  constexpr Tup u{u0, u1, u2};

  // Test whether the given `axis` is a separating axis (SA).
  auto isSA = [&](const auto &axis) {
    std::array p{Dot(ToTec(v[0]), axis), //
                 Dot(ToTec(v[1]), axis), //
                 Dot(ToTec(v[2]), axis)};

    auto abs = [](const auto &v) { return v < 0 ? -v : v; };
    T r = Dot(ToTec(e), Tec{abs(Dot(u0, axis)), //
                            abs(Dot(u1, axis)), //
                            abs(Dot(u2, axis))});

    return std::max({p[0], p[1], p[2]}) < -r || std::min({p[0], p[1], p[2]}) > r;
  };

  // Whether the two primitives are separating.
  bool isS = false;
  // Test 9 axes given by the cross products of combination of edges from both.
  ForEach<3>([&]<auto iU>() { ForEach<3>([&]<auto iF>() { isS = isS || isSA(Cross(get<iU>(u), ToTec(f[iF]))); }); });
  // Test 3 face normals from the AABB.
  isS = isS || isSA(u0) || isSA(u1) || isSA(u2);
  // Test 1 face normal from the triangle.
  isS = isS || isSA(ToTec(f[0].cross(f[1])));

  return !isS;
}

} // namespace collision_detection::detail

} // namespace ARIA
