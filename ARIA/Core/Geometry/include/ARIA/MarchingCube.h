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
/// 1. The input is exactly ONE line, square, or cube.
/// 2. The output is the points, lines, or triangles extracted from the input.
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
      if constexpr (dim == 1) {
        //! `int8` is used here because it can be implicitly converted to `int`, `uint`, ...
        auto access = [&](int8 i) {
          if constexpr (is_invocable_with_brackets_v<decltype(accessor), int8> ||
                        std::is_invocable_v<decltype(accessor), int8>)
            // Try `accessor[0]` and `accessor(0)`.
            return invoke_with_brackets_or_parentheses(accessor, i);
          else
            // Try `accessor[Tup{0}]` and `accessor(Tup{0})`.
            return invoke_with_brackets_or_parentheses(accessor, Tup{i});
        };
        // Rearrange and linearize into a `cuda::std::array`.
        using T = decltype(Auto(access(0)));
        return cuda::std::array<T, 2>{access(0), access(1)};
      } else if constexpr (dim == 2) {
        auto access = [&](int8 i, int8 j) {
          if constexpr (is_invocable_with_brackets_v<decltype(accessor), int8, int8> ||
                        std::is_invocable_v<decltype(accessor), int8, int8>)
            return invoke_with_brackets_or_parentheses(accessor, i, j);
          else
            return invoke_with_brackets_or_parentheses(accessor, Tup{i, j});
        };
        using T = decltype(Auto(access(0, 0)));
        return cuda::std::array<T, 4>{access(0, 0), access(1, 0), access(1, 1), access(0, 1)};
      } else if constexpr (dim == 3) {
        auto access = [&](int8 i, int8 j, int8 k) {
          if constexpr (is_invocable_with_brackets_v<decltype(accessor), int8, int8, int8> ||
                        std::is_invocable_v<decltype(accessor), int8, int8, int8>)
            return invoke_with_brackets_or_parentheses(accessor, i, j, k);
          else
            return invoke_with_brackets_or_parentheses(accessor, Tup{i, j, k});
        };
        using T = decltype(Auto(access(0, 0, 0)));
        return cuda::std::array<T, 8>{access(0, 0, 0), access(1, 0, 0), access(1, 1, 0), access(0, 1, 0), //
                                      access(0, 0, 1), access(1, 0, 1), access(1, 1, 1), access(0, 1, 1)};
      }
    };
    cuda::std::array positions = rearrangeAndLinearize(std::forward<TAccessorPositions>(accessorPositions));
    cuda::std::array values = rearrangeAndLinearize(std::forward<TAccessorValues>(accessorValues));
    static_assert(std::is_same_v<decltype(positions), cuda::std::array<VecDr, pow<dim>(2)>>,
                  "Invalid accessor of positions");
    static_assert(std::is_same_v<decltype(values), cuda::std::array<Real, pow<dim>(2)>>, "Invalid accessor of values");

    uint iCases = 0;
    ForEach<pow<dim>(2)>([&](auto i) {
      if (values[i] > isoValue)
        iCases += (1 << i);
    });
    if (iCases == 0 || iCases == pow<pow<dim - 1>(2)>(4) - 1)
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
