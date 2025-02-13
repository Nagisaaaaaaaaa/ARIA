#pragma once

/// \file
/// \brief A triangle implementation.
///
/// `Triangle` is implemented generically in dimension, thus can support
/// physics systems with any dimensions, 1D, 2D, 3D, 4D, ...

//
//
//
//
//
#include "ARIA/Vec.h"

namespace ARIA {

/// \brief A triangle implementation.
///
/// \tparam d Dimension.
///
/// \example ```cpp
/// // Create a `Triangle`.
/// Triangle3r tri{{0_R, 0_R, 0_R}, {1_R, 0_R, 0_R}, {0_R, 1_R, 0_R}};
///
/// // Get the vertices and normal of the `Triangle`.
/// Vec3r v0 = tri[0], v1 = tri[1], v2 = tri[2];
/// Vec3r normal = tri.normal();
/// Vec3r stableNormal = tri.stableNormal();
/// ```
template <typename T, uint d>
class Triangle final {
private:
  using VecDT = Vec<T, d>;

public:
  Triangle() = default;

  ARIA_HOST_DEVICE /*constexpr*/ Triangle(const VecDT &v0, const VecDT &v1, const VecDT &v2) : vertices_{v0, v1, v2} {}

  ARIA_COPY_MOVE_ABILITY(Triangle, default, default);

public:
  [[nodiscard]] ARIA_HOST_DEVICE /*constexpr*/ const VecDT &operator[](size_t i) const { return vertices_[i]; }

  [[nodiscard]] ARIA_HOST_DEVICE /*constexpr*/ VecDT &operator[](size_t i) { return vertices_[i]; }

public:
  [[nodiscard]] ARIA_HOST_DEVICE /*constexpr*/ VecDT normal() const {
    VecDT normal = (operator[](1) - operator[](0)).cross(operator[](2) - operator[](0));
    return normal.normalized();
  }

  [[nodiscard]] ARIA_HOST_DEVICE /*constexpr*/ VecDT stableNormal() const {
    VecDT normal = (operator[](1) - operator[](0)).cross(operator[](2) - operator[](0));
    return normal.stableNormalized();
  }

private:
  std::array<VecDT, 3> vertices_;
};

//
//
//
//
//
template <typename T>
using Triangle1 = Triangle<T, 1>;
template <typename T>
using Triangle2 = Triangle<T, 2>;
template <typename T>
using Triangle3 = Triangle<T, 3>;
template <typename T>
using Triangle4 = Triangle<T, 4>;

using Triangle1i = Triangle1<int>;
using Triangle1u = Triangle1<uint>;
using Triangle1f = Triangle1<float>;
using Triangle1d = Triangle1<double>;
using Triangle1r = Triangle1<Real>;

using Triangle2i = Triangle2<int>;
using Triangle2u = Triangle2<uint>;
using Triangle2f = Triangle2<float>;
using Triangle2d = Triangle2<double>;
using Triangle2r = Triangle2<Real>;

using Triangle3i = Triangle3<int>;
using Triangle3u = Triangle3<uint>;
using Triangle3f = Triangle3<float>;
using Triangle3d = Triangle3<double>;
using Triangle3r = Triangle3<Real>;

using Triangle4i = Triangle4<int>;
using Triangle4u = Triangle4<uint>;
using Triangle4f = Triangle4<float>;
using Triangle4d = Triangle4<double>;
using Triangle4r = Triangle4<Real>;

} // namespace ARIA
