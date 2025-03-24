#pragma once

#include "ARIA/AABB.h"
#include "ARIA/LineSegment.h"
#include "ARIA/Triangle.h"

namespace ARIA {

namespace collision_detection::detail {

//! `Tec` will be intensively used to ensure that
//! expressions will be truly simplified at compile-time.

namespace D1 {

// Unit `Tec`s are commonly used in collision detection algorithms.
static ARIA_CONST constexpr Tec u0{1_I};
static ARIA_CONST constexpr Tup u{u0};

} // namespace D1

namespace D2 {

static ARIA_CONST constexpr Tec u0{1_I, 0_I};
static ARIA_CONST constexpr Tec u1{0_I, 1_I};
static ARIA_CONST constexpr Tup u{u0, u1};

} // namespace D2

namespace D3 {

static ARIA_CONST constexpr Tec u0{1_I, 0_I, 0_I};
static ARIA_CONST constexpr Tec u1{0_I, 1_I, 0_I};
static ARIA_CONST constexpr Tec u2{0_I, 0_I, 1_I};
static ARIA_CONST constexpr Tup u{u0, u1, u2};

} // namespace D3

//
//
//
// Test whether the given `axis` is a separating axis (SA) for:
// 1. An `AABB` with extent equals to `extent` and center located at origin.
// 2. A `prim` with `n` vertices.
template <typename T, uint d, uint n>
[[nodiscard]] ARIA_HOST_DEVICE bool
IsSAForAABB(const Vec<T, d> &extent, const std::array<Vec<T, d>, n> &prim, const auto &axis) {
  static_assert(d == 1 || d == 2 || d == 3, "Collision detection algorithms are undefined for the given dimension `d`");

  std::array<T, n> p;
  ForEach<n>([&](auto i) { p[i] = Dot(ToTec(prim[i]), axis); });

  auto abs = [](const auto &v) { return v < 0 ? -v : v; };
  T r;
  if constexpr (d == 1) {
    using namespace D1;
    r = Dot(ToTec(extent), Tec{abs(Dot(u0, axis))});
  } else if constexpr (d == 2) {
    using namespace D2;
    r = Dot(ToTec(extent), Tec{abs(Dot(u0, axis)), //
                               abs(Dot(u1, axis))});
  } else if constexpr (d == 3) {
    using namespace D3;
    r = Dot(ToTec(extent), Tec{abs(Dot(u0, axis)), //
                               abs(Dot(u1, axis)), //
                               abs(Dot(u2, axis))});
  }

  T max = p[0], min = p[0];
  ForEach<n>([&](auto i) {
    if constexpr (i == 0)
      return;
    if (max < p[i])
      max = p[i];
    if (min > p[i])
      min = p[i];
  });

  return max < -r || min > r;
}

//
//
//
//
//
//! The following SAT algorithms are implemented based on
//! 2005, Christer Ericson, Real-Time Collision Detection, Chapter 5.2.9 Testing AABB Against Triangle.

template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool SAT(const AABB2<T> &aabb, const LineSegment2<T> &seg) {
  using namespace D2;
  using Vec2T = Vec2<T>;

  Vec2T c = aabb.center();
  Vec2T e = aabb.diagonal() / 2_R;

  std::array<Vec2T, 2> v;
  ForEach<2>([&]<auto i>() { v[i] = seg[i] - c; });
  auto isSA = [&](const auto &axis) { return IsSAForAABB(e, v, axis); };

  Vec2T f = v[1] - v[0];

  // Whether the two primitives are separating.
  bool isS = false;
  // Test 2 edge normals from the AABB.
  isS = isS || isSA(u0) || isSA(u1);
  // Test 1 edge normal from the line segment.
  isS = isS || isSA(Tec{-f.y(), f.x()});

  return !isS;
}

template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool SAT(const AABB2<T> &aabb, const Triangle2<T> &tri) {
  using namespace D2;
  using Vec2T = Vec2<T>;

  Vec2T c = aabb.center();
  Vec2T e = aabb.diagonal() / 2_R;

  std::array<Vec2T, 3> v;
  ForEach<3>([&]<auto i>() { v[i] = tri[i] - c; });
  auto isSA = [&](const auto &axis) { return IsSAForAABB(e, v, axis); };

  std::array<Vec2T, 3> f;
  ForEach<3>([&]<auto i>() { f[i] = v[(i + 1) % 3] - v[i]; });

  // Whether the two primitives are separating.
  bool isS = false;
  // Test 2 edge normals from the AABB.
  isS = isS || isSA(u0) || isSA(u1);
  // Test 3 edge normals from the triangle.
  ForEach<3>([&]<auto i>() { isS = isS || isSA(Tec{-f[i].y(), f[i].x()}); });

  return !isS;
}

//
//
//
template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool SAT(const AABB3<T> &aabb, const Triangle3<T> &tri) {
  using namespace D3;
  using Vec3T = Vec3<T>;

  Vec3T c = aabb.center();
  Vec3T e = aabb.diagonal() / 2_R;

  std::array<Vec3T, 3> v;
  ForEach<3>([&]<auto i>() { v[i] = tri[i] - c; });
  auto isSA = [&](const auto &axis) { return IsSAForAABB(e, v, axis); };

  std::array<Vec3T, 3> f;
  ForEach<3>([&]<auto i>() { f[i] = v[(i + 1) % 3] - v[i]; });

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
