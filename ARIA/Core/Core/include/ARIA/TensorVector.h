#pragma once

/// \file
/// \brief A multidimensional vector implementation based on NVIDIA CuTe.
/// See https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/03_tensor.md.
///
/// CuTe's `Tensor` class represents a multidimensional vector.
/// The vector's elements can live in any kind of memory, including
/// global memory, shared memory, and register memory.
///
/// While `Tensor`s can be either owning or non-owning,
/// `TensorVector`s are owning containers which can generate non-owning `Tensor`s.
/// So, `TensorVector`s can be seen as multidimensional
/// `thrust::host_vector`s or `thrust::device_vector`s.
///
/// \note We assume users have basic knowledge about CuTe.
/// If you are not familiar with CuTe, please read the tutorials of NVIDIA CuTe before continue:
/// 0. https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/00_quickstart.md.
/// 1. https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md.
/// 2. ...
///
/// \warning CuTe only support `int` as indices, so does `TensorVector`.
/// So, pay attention to large `TensorVector`s.

//
//
//
//
//
#include "ARIA/detail/TensorVectorImpl.h"

namespace ARIA {

/// \brief A `TensorVector` is a multidimensional vector, can be seen as a
/// multidimensional `thrust::host_vector` or `thrust::device_vector` with a `Layout`.
/// `TensorVector`s can generate `Tensor`s (views) which can be used at host or device side.
///
/// \example ```cpp
/// // Initialize an 1D `TensorVector`:
/// TensorVector<float, SpaceHost> v;   // An 1D host vector containing `float`,
///                                     // similar to `std::vector<float>`.
/// TensorVectorHost<float> v;          // The same.
///
/// TensorVector<float, SpaceDevice> v; // An 1D device vector containing `float`,
///                                     // similar to `thrust::device_vector<float>`.
/// TensorVectorDevice<float> v;        // The same.
///
/// v.Realloc(make_layout_major(100));  // Reallocate to size 100.
///
/// auto v = make_tensor_vector<float, SpaceHost>(make_layout_major(100)); // Make. (`Auto` not required here.)
///
/// auto myLayout = make_layout(...);                        // Make a tailored layout.
/// using MyLayout = decltype(myLayout);
///
/// TensorVector<float, SpaceHost, MyLayout> v;              // Use tailored layout.
/// TensorVectorHost<float, MyLayout> v;                     // The same.
///
/// auto v = make_tensor_vector<float, SpaceHost>(myLayout); // The same.
///
/// using TheDeviceVersion = TensorVectorHost<float>::template Mirrored<SpaceDevice>; // Only change the space.
///
///
///
/// // Initialize a 2D `TensorVector`:
/// TensorVector<float, C<2>, SpaceHost> v;     // A 2D left-layout (by default) host vector containing `float`.
/// TensorVectorHost<float, C<2>> v;            // The same.
/// TensorVector<float, C<2>, SpaceDevice> v;   // A 2D left-layout (by default) device vector containing `float`.
/// TensorVectorDevice<float, C<2>> v;          // The same.
///
/// v.Realloc(make_layout_major(10, 10));       // Reallocate to size 10 * 10.
///
/// auto v = make_tensor_vector<float, SpaceHost>(make_layout_major(10, 10)); // Make. (`Auto` not required here.)
///
/// auto myLayout = make_layout(...);                        // Make a tailored layout.
/// using MyLayout = decltype(myLayout);
///
/// TensorVector<float, C<2>, SpaceHost, MyLayout> v;        // Use tailored layout.
/// TensorVectorHost<float, C<2>, MyLayout> v;               // The same.
///
/// auto v = make_tensor_vector<float, SpaceHost>(myLayout); // The same.
///
/// using TheDeviceVersion = TensorVectorHost<float, C<2>>::template Mirrored<SpaceDevice>; // Only change the space.
///
///
///
/// // Usage:
///
/// // For both 1D and 2D.
/// for (int i = 0; i < v.size(); ++i)
///   v(i) = ...;
///
/// // For 2D only.
/// for (int y = 0; y < v.size<1>(); ++y)
///   for (int x = 0; x < v.size<0>(); ++x)
///     v(x, y) = ...;
///
/// // Note, you can unroll the loops at compile time if
/// // `v.size()` or `v.size<i>()` is a compile-time constant integral.
///
/// // Copy.
/// TensorVectorDevice<float> dst;
/// TensorVectorHost<float> src;
/// copy(dst, src); // Copy all the elements from `src` to `dst`.
///
/// // Tensor.
/// dst.tensor(); // Generate a non-owning tensor.
///               // For host `TensorVector`s, it is able to access elements at host side.
///               // For device `TensorVector`s, it is able to access elements at BOTH host and device side.
///
/// SomeKernel<<<1, 1>>>(dst.tensor());
/// ```
///
/// \warning CuTe only support `int` as indices, so does `TensorVector`.
/// So, pay attention to large `TensorVector`s.
using tensor_vector::detail::TensorVector;

/// \brief A host `TensorVector`.
///
/// \see TensorVector
using tensor_vector::detail::TensorVectorHost;

/// \brief A device `TensorVector`.
///
/// \see TensorVector
using tensor_vector::detail::TensorVectorDevice;

//
//
//
/// \brief Create a `TensorVector`.
///
/// \see TensorVector
using tensor_vector::detail::make_tensor_vector;

//
//
//
/// \see is_tensor_vector_v
using tensor_vector::detail::is_tensor_vector;

/// \brief Whether the given type is a tensor vector.
using tensor_vector::detail::is_tensor_vector_v;

//
//
//
/// \brief Copy all the elements from `src` to `dst`.
/// Copy succeed only when `src` and `dst` have exactly the same layout.
///
/// \example ```
/// TensorVectorDevice<float> dst;
/// TensorVectorHost<float> src;
/// ...
/// copy(dst, src);
/// ```
///
/// \warning For host code, exceptions are always thrown if copy fails.
/// For device code, only assertions.
using tensor_vector::detail::copy;

} // namespace ARIA
