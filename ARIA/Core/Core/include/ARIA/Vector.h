#pragma once

#include "ARIA/detail/MosaicVector.h"

namespace ARIA {

/// struct Pattern {
///   struct {
///     int v0;
///     int v1;
///   } v01;
///
///   int v2;
/// };
///
/// template <>
/// struct Mosaic<Tec<int, int, int>, Pattern> {
///   Pattern operator()(const Tec<int, int, int> &v) const {
///     return {.v01 = {.v0 = get<0>(v), .v1 = get<1>(v)}, .v2 = get<2>(v)};
///   }
///
///   Tec<int, int, int> operator()(const Pattern &v) const { return {v.v01.v0, v.v01.v1, v.v2}; }
/// };
///
/// using T = Tec<int, int, int>;
/// using TMosaic = Mosaic<T, Pattern>;
///
/// VectorHost<T> v0;         // Host   vector with array of structures (AoS).
/// VectorDevice<T> v1;       // Device vector with array of structures (AoS).
/// VectorHost<TMosaic> v2;   // Host   vector with structure of arrays (SoA).
/// VectorDevice<TMosaic> v3; // Device vector with structure of arrays (SoA).
template <typename T, typename TSpaceHostOrDevice, typename... Ts>
using Vector = mosaic::detail::reduce_vector_t<T, TSpaceHostOrDevice, Ts...>;

template <typename T, typename... Ts>
using VectorHost = Vector<T, SpaceHost, Ts...>;

template <typename T, typename... Ts>
using VectorDevice = Vector<T, SpaceDevice, Ts...>;

} // namespace ARIA
