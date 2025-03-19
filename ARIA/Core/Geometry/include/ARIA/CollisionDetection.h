#pragma once

#include "ARIA/AABB.h"
#include "ARIA/Triangle.h"

namespace ARIA {

template <typename T, uint d>
[[nodiscard]] ARIA_HOST_DEVICE bool SATImpl(const AABB<T, d> &aabb, const Triangle<T, d> &tri) {
  using VecDT = Vec<T, d>;

  VecDT c = aabb.center();
  VecDT e = aabb.diagonal() / 2_R;

  Triangle v = tri;
  ForEach<3>([&]<auto i>() { v[i] -= c; });

  std::array<VecDT, 3> f;
  ForEach<3>([&]<auto i>() { f[i] = v[(i + 1) % 3] - v[i]; });

  constexpr C<T(0)> _0;
  constexpr C<T(1)> _1;

  constexpr Tec u0{_1, _0, _0};
  constexpr Tec u1{_0, _1, _0};
  constexpr Tec u2{_0, _0, _1};
  constexpr Tup u{u0, u1, u2};
}

} // namespace ARIA
