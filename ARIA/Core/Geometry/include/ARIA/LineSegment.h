#pragma once

/// \file
/// \brief A line segment implementation.
///
/// `LineSegment` is implemented generically in dimension, thus can support
/// physics systems with any dimensions, 1D, 2D, 3D, 4D, ...

//
//
//
//
//
#include "ARIA/Vec.h"

namespace ARIA {

/// \brief A line segment implementation.
///
/// \tparam d Dimension.
///
/// \example ```cpp
/// // Create a `LineSegment`.
/// LineSegment2r seg{Vec2r{1_R, 0_R}, Vec2r{0_R, 1_R}};
///
/// // Get the vertices of the `LineSegment`.
/// Vec2r v0 = seg[0], v1 = seg[1];
/// ```
template <typename T, uint d>
class LineSegment final {
private:
  using VecDT = Vec<T, d>;

public:
  LineSegment() = default;

  ARIA_HOST_DEVICE /*constexpr*/ LineSegment(const VecDT &v0, const VecDT &v1) : vertices_{v0, v1} {}

  ARIA_COPY_MOVE_ABILITY(LineSegment, default, default);

public:
  [[nodiscard]] ARIA_HOST_DEVICE /*constexpr*/ const VecDT &operator[](size_t i) const { return vertices_[i]; }

  [[nodiscard]] ARIA_HOST_DEVICE /*constexpr*/ VecDT &operator[](size_t i) { return vertices_[i]; }

private:
  std::array<VecDT, 2> vertices_;
};

//
//
//
//
//
template <typename T>
using LineSegment1 = LineSegment<T, 1>;
template <typename T>
using LineSegment2 = LineSegment<T, 2>;
template <typename T>
using LineSegment3 = LineSegment<T, 3>;
template <typename T>
using LineSegment4 = LineSegment<T, 4>;

using LineSegment1i = LineSegment1<int>;
using LineSegment1u = LineSegment1<uint>;
using LineSegment1f = LineSegment1<float>;
using LineSegment1d = LineSegment1<double>;
using LineSegment1r = LineSegment1<Real>;

using LineSegment2i = LineSegment2<int>;
using LineSegment2u = LineSegment2<uint>;
using LineSegment2f = LineSegment2<float>;
using LineSegment2d = LineSegment2<double>;
using LineSegment2r = LineSegment2<Real>;

using LineSegment3i = LineSegment3<int>;
using LineSegment3u = LineSegment3<uint>;
using LineSegment3f = LineSegment3<float>;
using LineSegment3d = LineSegment3<double>;
using LineSegment3r = LineSegment3<Real>;

using LineSegment4i = LineSegment4<int>;
using LineSegment4u = LineSegment4<uint>;
using LineSegment4f = LineSegment4<float>;
using LineSegment4d = LineSegment4<double>;
using LineSegment4r = LineSegment4<Real>;

} // namespace ARIA
