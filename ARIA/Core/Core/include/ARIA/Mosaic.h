#pragma once

/// \file
/// \brief `Mosaic` is an abstraction about how to describe one type with
/// another type, where the later type can be easily and automatically serialized.
///
/// For example, `Vec3f` might be extremely complex, but
/// `struct Pattern { float x, y, z; };` is always simple.
///
/// Suppose ARIA knows how to convert `Vec3f` to and from `Pattern`,
/// then, ARIA can automatically do a lot of things for us.
/// For example, arrays and vectors of `Vec3f` can be automatically
/// converted to structure-of-arrays (SoA) storages, etc.
///
/// Here lists all the ARIA built-in features which
/// are compatible with `Mosaic`:
/// 1. See `Array.h`.
/// 2. See `TensorVector.h`.
/// 3. See `VDB.h`.
/// 4. See `Vector.h`.
///
/// Users only need to define some simple types and methods,
/// see `class Mosaic` below, and all things will be ready.
///
/// \details Actually, it is better to use the name `Puzzle`, but
/// the main developer of ARIA really likes "Kin-iro Mosaic".
/// That's why `Mosaic` is used.

//
//
//
//
//
#include "ARIA/TypeArray.h"

#include <boost/pfr.hpp>

namespace ARIA {

/// \brief `Mosaic` defines how to convert `T` to and from `TMosaicPattern`, where
/// the later type can be easily and automatically serialized.
///
/// \example ```cpp
/// // Define a mosaic pattern for `Vec3<T>`.
/// template <typename T>
/// struct Pattern {
///   T x, y, z; // `T values[3]` is also allowed here.
/// };
/// // You can similarly define a more generic pattern for `Vec<T, size>`.
///
/// // Define the mosaic for `Vec3<T>` and `Pattern<T>`.
/// template <typename T>
/// struct Mosaic<Vec3<T>, Pattern<T>> {
///   // How to convert `Vec3<T>` to `Pattern<T>`.
///   Pattern<T> operator()(const Vec3<T> &v) const { return {.x = v.x(), .y = v.y(), .z = v.z()}; }
///
///   // How to convert `Pattern<T>` to `Vec3<T>`.
///   Vec3<T> operator()(const Pattern<T> &v) const { return {v.x, v.y, v.z}; }
/// };
///
/// // Define the mosaic for `double` and `float`.
/// template <>
/// struct Mosaic<double, float> {
///   // How to convert `double` to `float`.
///   float operator()(double v) const { return v; }
///
///   // How to convert `float` to `double`.
///   double operator()(float v) const { return v; }
/// };
///
/// // Define the mosaic for `float` and `float`.
/// template <>
/// struct Mosaic<float, float> {
///   // Check whether the floating point is still valid.
///   float operator()(float v) const {
///     ARIA_ASSERT(!std::isnan(v));
///     return v;
///   }
/// };
/// ```
///
/// \note Very complex pattern structures are also supported, but
/// member variables should be either "scalar" or "aggregate".
/// See `std::scalar` and `std::aggregate`.
///
/// \see Array.h
/// \see TensorVector.h
/// \see VDB.h
/// \see Vector.h
template <typename T, typename TMosaicPattern>
class Mosaic;

//
//
//
//
//
namespace mosaic::detail {

template <typename T>
struct MosaicBuiltIn;

} // namespace mosaic::detail

/// \brief Shortcuts to the ARIA built-in `Mosaic`s, which are listed here:
/// 1. See `Mat.h`.
/// 2. See `Quat.h`.
/// 3. See `Vec.h`.
///
/// \example ```cpp
/// using TMosaic = mosaic_t<Vec3r>;
/// ```
template <typename T>
using mosaic_t = typename mosaic::detail::MosaicBuiltIn<T>::type;

} // namespace ARIA

//
//
//
//
//
#include "ARIA/detail/Mosaic.inc"
