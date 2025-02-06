#pragma once

/// \file
/// \brief A fixed-size array implementation, which supports both
/// array of structures (AoS) and structure of arrays (SoA).
/// Please read the comments in `Mosaic.h` before continue.

//
//
//
//
//
#include "ARIA/detail/MosaicArray.h"

namespace ARIA {

/// \brief A fixed-size array implementation, which supports both
/// array of structures (AoS) and structure of arrays (SoA).
///
/// Interfaces are similar to `cuda::std::array`.
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
/// using T = Vec3<int>;
/// using TMosaic = Mosaic<T, Pattern<int>>;
///
/// Array<T, 10> v0;         // Fixed-size array with array of structures (AoS), that is, `array<Vec3<int>, 10>`.
/// Array<TMosaic, 10> v1;   // Fixed-size array with structure of arrays (SoA), that is, `3 * array<int, 10>`.
/// ```
///
/// \warning Just like `std::vector<bool>`, `operator[]`s of SoA variants
/// will return proxies instead of simple references, see `Auto.h`.
template <typename T, size_t size>
using Array = mosaic::detail::reduce_array_t<T, size>;

} // namespace ARIA
