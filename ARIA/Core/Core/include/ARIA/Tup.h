#pragma once

/// \file
/// \brief TODO: Document this.

//
//
//
//
//
#include "ARIA/detail/TupImpl.h"

namespace ARIA {

/// \brief `Tup` (tuple) is a fixed-size collection of heterogeneous values.
///
/// \example ```cpp
/// Tup tup{1, C<2U>{}, Tup{3.0F, std::string{"4"}}};
/// ```
///
/// \warning `Tup` has the same type as `cute::tuple`.
using tup::detail::Tup;

/// \brief `Tec` (tuple + vec) is a fixed-size collection of heterogeneous "arithmetic" values.
///
/// \example ```cpp
/// Tec tec{1, C<2U>{}, 3.0F, C<4.0>{}};
/// Tec3 tec3{1, C<2U>{}, 3.0F};
/// Teci teci{1, C<2>{}, 3, C<4>{}};
/// Tec3i tec3i{1, C<2>{}, 3};
/// ```
///
/// \warning `Tec` has the same type as `cute::tuple` but with restrictions, where
/// the elements are required to be "arithmetic" types.
using tup::detail::Tec;

using tup::detail::Tec1;
using tup::detail::Tec2;
using tup::detail::Tec3;
using tup::detail::Tec4;

using tup::detail::Tecd;
using tup::detail::Tecf;
using tup::detail::Teci;
using tup::detail::Tecr;
using tup::detail::Tecu;

using tup::detail::Tec1d;
using tup::detail::Tec1f;
using tup::detail::Tec1i;
using tup::detail::Tec1r;
using tup::detail::Tec1u;

using tup::detail::Tec2d;
using tup::detail::Tec2f;
using tup::detail::Tec2i;
using tup::detail::Tec2r;
using tup::detail::Tec2u;

using tup::detail::Tec3d;
using tup::detail::Tec3f;
using tup::detail::Tec3i;
using tup::detail::Tec3r;
using tup::detail::Tec3u;

using tup::detail::Tec4d;
using tup::detail::Tec4f;
using tup::detail::Tec4i;
using tup::detail::Tec4r;
using tup::detail::Tec4u;

//
//
//
//
//
using tup::detail::tup_size_v;

using tup::detail::tup_elem_t;

//
//
//
using tup::detail::rank;

using tup::detail::rank_t;

using tup::detail::rank_v;

using tup::detail::get;

//
//
//
using tup::detail::is_static;

using tup::detail::is_static_v;

//
//
//
//
//
/// \brief Cast `Tec` to `std::array`.
///
/// \example ```cpp
/// std::array<int, 2> array0 = ToArray(Tec{5, 6});
/// std::array<int, 2> array1 = ToArray(Tec{5_I, 6});
/// std::array<int, 2> array2 = ToArray(Tec{5_I, 6_I});
/// ```
using tup::detail::ToArray;

} // namespace ARIA
