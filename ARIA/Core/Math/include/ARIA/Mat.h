#pragma once

/// \file
/// \brief ARIA directly uses `Eigen` as linear algebra library.
/// Make sure you are familiar with `Eigen` before continue.
/// See https://eigen.tuxfamily.org.
///
/// A `Mat` is a direct using of the `Eigen::Matrix`.

//
//
//
//
//
#include "ARIA/detail/MatImpl.h"

namespace ARIA {

/// \brief A direct using of the `Eigen::Matrix`.
template <typename T, auto row, auto col>
using Mat = mat::detail::Mat<T, row, col>;

//
//
//
template <typename T>
using Mat2 = Mat<T, 2, 2>;
template <typename T>
using Mat3 = Mat<T, 3, 3>;
template <typename T>
using Mat4 = Mat<T, 4, 4>;

using Mat2i = Mat2<int>;
using Mat2u = Mat2<uint>;
using Mat2f = Mat2<float>;
using Mat2d = Mat2<double>;
using Mat2r = Mat2<Real>;

using Mat3i = Mat3<int>;
using Mat3u = Mat3<uint>;
using Mat3f = Mat3<float>;
using Mat3d = Mat3<double>;
using Mat3r = Mat3<Real>;

using Mat4i = Mat4<int>;
using Mat4u = Mat4<uint>;
using Mat4f = Mat4<float>;
using Mat4d = Mat4<double>;
using Mat4r = Mat4<Real>;

//
//
//
//
//
/// \brief A property prefab for `class Mat`.
/// All possible sub-properties and functions have been defined here.
///
/// \example ```cpp
/// ARIA_PROP_PREFAB_MAT(public, public, __host__, Mat3r, rotationMat)
/// ```
#define ARIA_PROP_PREFAB_MAT(accessGet, accessSet, specifiers, type, propName)                                         \
  __ARIA_PROP_PREFAB_MAT(accessGet, accessSet, specifiers, type, propName)

/// \brief A sub-property prefab for `class Mat`.
/// All possible sub-sub-properties and functions have been defined here.
///
/// \example ```cpp
/// ARIA_SUB_PROP_PREFAB_MAT(__host__, Mat3r, rotationMat)
/// ```
#define ARIA_SUB_PROP_PREFAB_MAT(specifiers, type, propName) __ARIA_SUB_PROP_PREFAB_MAT(specifiers, type, propName)

} // namespace ARIA
