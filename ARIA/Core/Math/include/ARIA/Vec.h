#pragma once

/// \file
/// \brief ARIA directly uses `Eigen` as linear algebra library.
/// Make sure you are familiar with `Eigen` before continue.
/// See https://eigen.tuxfamily.org.
///
/// A `Vec` is a direct using of the `Eigen::Vector`.

//
//
//
//
//
#include "ARIA/detail/VecImpl.h"

namespace ARIA {

/// \brief A direct using of the `Eigen::Vector`.
template <typename T, size_t s>
using Vec = Eigen::Vector<T, s>;

//
//
//
template <typename T>
using Vec1 = Vec<T, 1>;
template <typename T>
using Vec2 = Vec<T, 2>;
template <typename T>
using Vec3 = Vec<T, 3>;
template <typename T>
using Vec4 = Vec<T, 4>;

using Vec1i = Vec1<int>;
using Vec1u = Vec1<uint>;
using Vec1f = Vec1<float>;
using Vec1d = Vec1<double>;
using Vec1r = Vec1<Real>;

using Vec2i = Vec2<int>;
using Vec2u = Vec2<uint>;
using Vec2f = Vec2<float>;
using Vec2d = Vec2<double>;
using Vec2r = Vec2<Real>;

using Vec3i = Vec3<int>;
using Vec3u = Vec3<uint>;
using Vec3f = Vec3<float>;
using Vec3d = Vec3<double>;
using Vec3r = Vec3<Real>;

using Vec4i = Vec4<int>;
using Vec4u = Vec4<uint>;
using Vec4f = Vec4<float>;
using Vec4d = Vec4<double>;
using Vec4r = Vec4<Real>;

//
//
//
//
//
/// \brief A property prefab for `class Vec`.
/// All possible sub-properties and functions have been defined here.
///
/// \example ```cpp
/// ARIA_PROP_PREFAB_VEC(public, public, __host__, Vec3r, position)
/// ```
#define ARIA_PROP_PREFAB_VEC /*(accessGet, accessSet, specifiers, type, propName, (optional) args...)*/                \
  __ARIA_PROP_PREFAB_VEC

/// \brief A sub-property prefab for `class Vec`.
/// All possible sub-sub-properties and functions have been defined here.
///
/// \example ```cpp
/// ARIA_SUB_PROP_PREFAB_VEC(__host__ , Vec3r, position)
/// ```
#define ARIA_SUB_PROP_PREFAB_VEC(specifiers, type, propName) __ARIA_SUB_PROP_PREFAB_VEC(specifiers, type, propName)

//
//
//
//
//
/// \brief Cast `Vec` to `Coord`.
///
/// \example ```cpp
/// Coord<int, int> coord = ToCoord(Vec2i{5, 6});
/// ```
using vec::detail::ToCoord;

/// \brief Cast `Coord` to `Vec`.
///
/// \example ```cpp
/// Vec2i vec0 = ToVec(make_coord(5, 6));
/// Vec2i vec1 = ToVec(make_coord(5_I, 6));
/// Vec2i vec2 = ToVec(make_coord(5_I, 6_I));
/// ```
using vec::detail::ToVec;

} // namespace ARIA
