#pragma once

/// \file
/// \brief A tuple abstraction based on NVIDIA CuTe.
/// See https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md.
///
/// CuTe is a collection of C++ CUDA template abstractions for
/// defining and operating on hierarchically multidimensional layouts of threads and data.
/// CuTe provides `Layout` and `Tensor` objects that
/// compactly packages the type, shape, memory space, and layout of data, while
/// performing the complicated indexing for the user.
/// This lets programmers focus on the logical descriptions of their algorithms while
/// CuTe does the mechanical bookkeeping for them.
/// With these tools, we can quickly design, implement, and modify all dense linear algebra operations.
///
/// The core abstraction of CuTe are the hierarchically multidimensional layouts which
/// can be composed with data arrays to represent tensors.
/// The representation of layouts is powerful enough to represent nearly everything
/// we need to implement efficient dense linear algebra.
/// Layouts can also be combined and manipulated via functional composition, on which
/// we build a large set of common operations such as tiling and partitioning.
///
/// This file introduces `Tup` (tuple) and `Tec` (tuple + vec), where
/// `Tec` is a vec type which can be fully or even partially determined at compile time.
/// `Tup` and `Tec` are extensively used in ARIA metaprogramming.
///
/// \note This file is not fully documented, since we assume that the users have basic knowledge about CuTe.
/// If you are not familiar with CuTe, please read the tutorials of NVIDIA CuTe before continue:
/// 0. https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md.
/// 1. https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md.
/// 2. ...

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
///
/// Tec3i a = {1, 2_I, 3};
/// Tec3i b = {0, 2_I, 4};
/// Tec3i c = a + b; // `c` contains `1`, `4_I`, `7`.
/// ```
///
/// \warning `Tec` has the same type as `cute::tuple` but with restrictions, where
/// the elements are required to be "arithmetic" types.
/// Here, "arithmetic" means `std::is_arithmetic` or `ConstantArithmetic`.
using tup::detail::Tec;

//
//
//
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
/// \brief Cast `Tup` to `TypeArray`.
///
/// \example ```cpp
/// Tup tup{1, C<2U>{}, Tup{3.0F, std::string{"4"}}};
/// using Ts = to_type_array_t<decltype(tup)>; // TypeArray<int, C<2U>, Tup<float, std::string>>
/// ```
using tup::detail::to_type_array_t;

/// \brief Cast `TypeArray` to `Tup`.
///
/// \example ```cpp
/// using Ts = MakeTypeArray<int, C<2U>, Tup<float, std::string>>;
/// using TTup = to_tup_t<Ts>; // Tup<int, C<2U>, Tup<float, std::string>>
/// ```
using tup::detail::to_tup_t;

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
