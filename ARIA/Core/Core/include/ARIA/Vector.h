#pragma once

#include "ARIA/detail/MosaicVector.h"

namespace ARIA {

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
/// VectorHost<T> v0;         // Host   vector with array of structures (AoS).
/// VectorDevice<T> v1;       // Device vector with array of structures (AoS).
/// VectorHost<TMosaic> v2;   // Host   vector with structure of arrays (SoA).
/// VectorDevice<TMosaic> v3; // Device vector with structure of arrays (SoA).
/// ```
template <typename T, typename TSpaceHostOrDevice, typename... Ts>
using Vector = mosaic::detail::reduce_vector_t<T, TSpaceHostOrDevice, Ts...>;

template <typename T, typename... Ts>
using VectorHost = Vector<T, SpaceHost, Ts...>;

template <typename T, typename... Ts>
using VectorDevice = Vector<T, SpaceDevice, Ts...>;

} // namespace ARIA
