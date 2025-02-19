#pragma once

/// \file
/// \brief A policy-based marching cube implementation.
///
/// Marching cubes is a computer graphics algorithm,
/// published in the 1987 SIGGRAPH proceedings by Lorensen and Cline, for extracting
/// a polygonal mesh of an iso-surface from a three-dimensional discrete scalar field.
/// See https://en.wikipedia.org/wiki/Marching_cubes.
///
/// This file introduces the 1D, 2D, and 3D versions of marching cube.
/// We say "cube" instead of "cubes" because:
/// 1. The input is exactly ONE line segment, square, or cube.
/// 2. The output is the points, line segments, or triangles extracted from the input.
///
/// Users can implement their own marching "cubes" based on it.

//
//
//
//
//
#include "ARIA/Invocations.h"
#include "ARIA/Math.h"
#include "ARIA/Vec.h"
#include "ARIA/detail/MarchingCubeConstants.h"

#include <cuda/std/array>

namespace ARIA {

/// \brief A policy-based marching cube implementation.
///
/// Marching cubes is a computer graphics algorithm,
/// published in the 1987 SIGGRAPH proceedings by Lorensen and Cline, for extracting
/// a polygonal mesh of an iso-surface from a three-dimensional discrete scalar field.
/// See https://en.wikipedia.org/wiki/Marching_cubes.
///
/// \tparam dim Dimension of the system.
///
/// \example ```cpp
/// using MC = MarchingCube<2>; // This example is in 2D.
///
/// // The positions of the 4 dual vertices, that is, the 4 vertices of the input square.
/// // `positions` will only be invoked with `(0, 0)`, `(1, 0)`, `(0, 1)` and `(1, 1)`.
/// auto positions = [&](uint x, uint y) -> Vec2r { ... };
/// // The values of the 4 dual vertices.
/// // `values` will only be invoked with `(0, 0)`, `(1, 0)`, `(0, 1)` and `(1, 1)`.
/// auto values = [&](uint x, uint y) -> Real { ... };
/// // For `positions` and `values`, the parameter type can also be something like
/// // `T, U`, `Tup<T, U>`, `C<i>, C<j>`, and `Tup<C<i>, C<j>>`, where
/// // `T, U` should be `<typename T, typename U>`, and `i, j` be `<auto i, auto j>`.
/// Real isoValue = 0.5_R;
///
/// // Extract line segments from the 4 dual vertices with the given iso-value.
/// // We will get 0, 1, or 2 line segments.
/// MC::Extract(positions, values, isoValue, [](cuda::std::span<Vec2r, 2> vertices) {
///   // Get a line segment with endpoints `v0` and `v1`.
///   // For cases where there are multiple line segments,
///   // this lambda function will be called multiple times.
///   Vec2r v0 = vertices[0];
///   Vec2r v1 = vertices[1];
/// });
/// ```
template <uint dim>
class MarchingCube {
public:
  template <typename TAccessorPositions, typename TAccessorValues, typename F>
  ARIA_HOST_DEVICE static constexpr void
  Extract(TAccessorPositions &&accessorPositions, TAccessorValues &&accessorValues, Real isoValue, F &&f) {
    //! The implementation is mainly based on https://github.com/ingowald/cudaAmrIsoSurfaceExtraction.
    //! It is recommended to read the paper before continue.
    using VecDr = Vec<Real, dim>;

    // VTK case tables assume everything is in VTK "hexahedron" ordering.
    // So, dual cells are rearranged and linearized into arrays.
    //! In order to support various accessors such as `accessor(0, 1)` and `accessor(Tup{0, 1})`,
    //! we will try and invoke the accessors with difference kinds of parameters.
    auto rearrangeAndLinearize = []<typename TAccessor>(TAccessor &&accessor) {
      //! `int8` is used here because it can be implicitly converted to `int`, `uint`, ...
      constexpr C<int8{0}> _0;
      constexpr C<int8{1}> _1;

      if constexpr (dim == 1) {
        auto access = [&]<int8 i>(C<i>) {
          if constexpr (is_invocable_with_brackets_v<decltype(accessor), C<i>> ||
                        std::is_invocable_v<decltype(accessor), C<i>>)
            // Try `accessor[0_I]` and `accessor(0_I)`.
            return invoke_with_brackets_or_parentheses(accessor, C<i>{});
          else if constexpr (is_invocable_with_brackets_v<decltype(accessor), Tup<C<i>>> ||
                             std::is_invocable_v<decltype(accessor), Tup<C<i>>>)
            // Try `accessor[Tup{0_I}]` and `accessor(Tup{0_I})`.
            return invoke_with_brackets_or_parentheses(accessor, Tup<C<i>>{});
          else if constexpr (is_invocable_with_brackets_v<decltype(accessor), int8> ||
                             std::is_invocable_v<decltype(accessor), int8>)
            // Try `accessor[0]` and `accessor(0)`.
            return invoke_with_brackets_or_parentheses(accessor, i);
          else
            // Try `accessor[Tup{0}]` and `accessor(Tup{0})`.
            return invoke_with_brackets_or_parentheses(accessor, Tup{i});
        };
        // Rearrange and linearize into a `cuda::std::array`.
        using T = decltype(Auto(access(_0)));
        return cuda::std::array<T, 2>{access(_0), access(_1)};
      } else if constexpr (dim == 2) {
        auto access = [&]<int8 i, int8 j>(C<i>, C<j>) {
          if constexpr (is_invocable_with_brackets_v<decltype(accessor), C<i>, C<j>> ||
                        std::is_invocable_v<decltype(accessor), C<i>, C<j>>)
            return invoke_with_brackets_or_parentheses(accessor, C<i>{}, C<j>{});
          else if constexpr (is_invocable_with_brackets_v<decltype(accessor), Tup<C<i>, C<j>>> ||
                             std::is_invocable_v<decltype(accessor), Tup<C<i>, C<j>>>)
            return invoke_with_brackets_or_parentheses(accessor, Tup<C<i>, C<j>>{});
          else if constexpr (is_invocable_with_brackets_v<decltype(accessor), int8, int8> ||
                             std::is_invocable_v<decltype(accessor), int8, int8>)
            return invoke_with_brackets_or_parentheses(accessor, i, j);
          else
            return invoke_with_brackets_or_parentheses(accessor, Tup{i, j});
        };
        using T = decltype(Auto(access(_0, _0)));
        return cuda::std::array<T, 4>{access(_0, _0), access(_1, _0), access(_1, _1), access(_0, _1)};
      } else if constexpr (dim == 3) {
        auto access = [&]<int8 i, int8 j, int8 k>(C<i>, C<j>, C<k>) {
          if constexpr (is_invocable_with_brackets_v<decltype(accessor), C<i>, C<j>, C<k>> ||
                        std::is_invocable_v<decltype(accessor), C<i>, C<j>, C<k>>)
            return invoke_with_brackets_or_parentheses(accessor, C<i>{}, C<j>{}, C<k>{});
          else if constexpr (is_invocable_with_brackets_v<decltype(accessor), Tup<C<i>, C<j>, C<k>>> ||
                             std::is_invocable_v<decltype(accessor), Tup<C<i>, C<j>, C<k>>>)
            return invoke_with_brackets_or_parentheses(accessor, Tup<C<i>, C<j>, C<k>>{});
          else if constexpr (is_invocable_with_brackets_v<decltype(accessor), int8, int8, int8> ||
                             std::is_invocable_v<decltype(accessor), int8, int8, int8>)
            return invoke_with_brackets_or_parentheses(accessor, i, j, k);
          else
            return invoke_with_brackets_or_parentheses(accessor, Tup{i, j, k});
        };
        using T = decltype(Auto(access(_0, _0, _0)));
        return cuda::std::array<T, 8>{access(_0, _0, _0), access(_1, _0, _0), access(_1, _1, _0), access(_0, _1, _0), //
                                      access(_0, _0, _1), access(_1, _0, _1), access(_1, _1, _1), access(_0, _1, _1)};
      }
    };
    cuda::std::array positions = rearrangeAndLinearize(std::forward<TAccessorPositions>(accessorPositions));
    cuda::std::array values = rearrangeAndLinearize(std::forward<TAccessorValues>(accessorValues));
    static_assert(std::is_same_v<decltype(positions), cuda::std::array<VecDr, Pow<dim>(2)>>,
                  "Invalid accessor of positions");
    static_assert(std::is_same_v<decltype(values), cuda::std::array<Real, Pow<dim>(2)>>, "Invalid accessor of values");

    uint iCases = 0;
    ForEach<Pow<dim>(2)>([&](auto i) {
      if (values[i] > isoValue)
        iCases += (1 << i);
    });
    if (iCases == 0 || iCases == Pow<Pow<dim - 1>(2)>(4) - 1)
      return;

    for (const int8_t *edge = marching_cube::detail::MarchingCubesCases<dim>()[iCases]; *edge > -1; edge += dim) {
      cuda::std::array<VecDr, dim> primitiveVertices;
      ForEach<dim>([&](auto i) {
        const int8_t *vert = marching_cube::detail::MarchingCubesEdges<dim>()[edge[i]];
        const VecDr p0 = positions[vert[0]];
        const VecDr p1 = positions[vert[1]];
        const Real v0 = values[vert[0]];
        const Real v1 = values[vert[1]];
        const Real t = (isoValue - v0) / (v1 - v0);
        primitiveVertices[i] = Lerp(p0, p1, t);
      });

      bool success = true;
      if constexpr (dim == 1) {
      } else if constexpr (dim == 2) {
        if (primitiveVertices[0] == primitiveVertices[1])
          success = false;
      } else if constexpr (dim == 3) {
        if (primitiveVertices[0] == primitiveVertices[1] || //
            primitiveVertices[1] == primitiveVertices[2] || //
            primitiveVertices[2] == primitiveVertices[0])
          success = false;
      }

      if (!success)
        continue;

      // Calls `f` with the primitive vertices.
      f(primitiveVertices);
    }
  }
};

} // namespace ARIA
