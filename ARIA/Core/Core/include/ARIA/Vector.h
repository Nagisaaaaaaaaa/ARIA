#pragma once

/// \file
/// \brief A policy-based vector implementation, which supports both
/// array of structures (AoS) and structure of arrays (SoA).
/// Please read the comments in `Mosaic.h` before continue.

//
//
//
//
//
#include "ARIA/detail/MosaicVector.h"

namespace ARIA {

/// \brief A policy-based vector implementation, which supports both
/// array of structures (AoS) and structure of arrays (SoA).
///
/// Interfaces are the same as `thrust::host_vector` and `thrust::device_vector`.
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
/// VectorHost<T> v0;         // Host   vector with array of structures (AoS), that is, `host_vector<Vec3<int>>`.
/// VectorDevice<T> v1;       // Device vector with array of structures (AoS), that is, `device_vector<Vec3<int>>`.
/// VectorHost<TMosaic> v2;   // Host   vector with structure of arrays (SoA), that is, `3 * host_vector<int>`.
/// VectorDevice<TMosaic> v3; // Device vector with structure of arrays (SoA), that is, `3 * device_vector<int>`.
/// ```
///
/// \warning Just like `thrust::device_vector`, `operator[]`s of SoA or device variants
/// will return proxies instead of simple references, see `Auto.h`.
template <typename T, typename TSpaceHostOrDevice, typename... Ts>
using Vector = mosaic::detail::reduce_vector_t<T, TSpaceHostOrDevice, Ts...>;

template <typename T, typename... Ts>
using VectorHost = Vector<T, SpaceHost, Ts...>;

template <typename T, typename... Ts>
using VectorDevice = Vector<T, SpaceDevice, Ts...>;

} // namespace ARIA
