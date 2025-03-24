#pragma once

#include "ARIA/AABB.h"
#include "ARIA/LineSegment.h"
#include "ARIA/Triangle.h"

namespace ARIA {

namespace collision_detection::detail {

//! `Tec` will be intensively used to ensure that
//! expressions will be truly simplified at compile-time.

namespace D2 {

// Unit `Tec`s are commonly used in collision detection algorithms.
static ARIA_CONST constexpr Tec u0{1_I, 0_I};
static ARIA_CONST constexpr Tec u1{0_I, 1_I};
static ARIA_CONST constexpr Tup u{u0, u1};

// Test whether the given `axis` is a separating axis (SA) for `prim` and
// an `AABB` with extent equals to `extent` and center located at origin.
template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool IsSAForAABB(const auto &extent, const auto &prim, const auto &axis) {
  std::array p{Dot(ToTec(prim[0]), axis), //
               Dot(ToTec(prim[1]), axis)};

  auto abs = [](const auto &v) { return v < 0 ? -v : v; };
  T r = Dot(ToTec(extent), Tec{abs(Dot(u0, axis)), //
                               abs(Dot(u1, axis))});

  return std::max(p[0], p[1]) < -r || std::min(p[0], p[1]) > r;
}

} // namespace D2

//
//
//
namespace D3 {

static ARIA_CONST constexpr Tec u0{1_I, 0_I, 0_I};
static ARIA_CONST constexpr Tec u1{0_I, 1_I, 0_I};
static ARIA_CONST constexpr Tec u2{0_I, 0_I, 1_I};
static ARIA_CONST constexpr Tup u{u0, u1, u2};

template <typename T>
[[nodiscard]] ARIA_HOST_DEVICE bool IsSAForAABB(const auto &extent, const auto &prim, const auto &axis) {
  std::array p{Dot(ToTec(prim[0]), axis), //
               Dot(ToTec(prim[1]), axis), //
               Dot(ToTec(prim[2]), axis)};

  auto abs = [](const auto &v) { return v < 0 ? -v : v; };
  T r = Dot(ToTec(extent), Tec{abs(Dot(u0, axis)), //
                               abs(Dot(u1, axis)), //
                               abs(Dot(u2, axis))});

  return std::max({p[0], p[1], p[2]}) < -r || std::min({p[0], p[1], p[2]}) > r;
}

} // namespace D3

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

  LineSegment v = seg;
  ForEach<2>([&]<auto i>() { v[i] -= c; });
  auto isSA = [&](const auto &axis) { return IsSAForAABB<T>(e, v, axis); };

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

  Triangle v = tri;
  ForEach<3>([&]<auto i>() { v[i] -= c; });
  auto isSA = [&](const auto &axis) { return IsSAForAABB<T>(e, v, axis); };

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

  Triangle v = tri;
  ForEach<3>([&]<auto i>() { v[i] -= c; });
  auto isSA = [&](const auto &axis) { return IsSAForAABB<T>(e, v, axis); };

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
