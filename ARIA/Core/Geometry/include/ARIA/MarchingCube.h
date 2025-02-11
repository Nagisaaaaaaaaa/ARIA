#pragma once

#include "ARIA/Math.h"
#include "ARIA/Vec.h"
#include "ARIA/detail/MarchingCubeConstants.h"

#include <cuda/std/array>

namespace ARIA {

template <uint dim>
class MarchingCubes {
public:
  template <typename TAccessorPositions, typename TAccessorValues, typename F>
  ARIA_HOST_DEVICE static constexpr void
  Extract(TAccessorPositions &&accessorPositions, TAccessorValues &&accessorValues, Real isoValue, F &&f) {
    //! VTK case tables assume everthing is in VTK "hexahedron" ordering.
    //! So, cells are rearranged and linearized into arrays.
    auto rearrangeAndLinearize = []<typename TAccessor>(TAccessor &&accessor) {
      if constexpr (dim == 1) {
        using T = decltype(Auto(accessor(0)));
        return cuda::std::array<T, 2>{accessor(0), accessor(1)};
      } else if constexpr (dim == 2) {
        using T = decltype(Auto(accessor(0, 0)));
        return cuda::std::array<T, 4>{accessor(0, 0), accessor(1, 0), accessor(1, 1), accessor(0, 1)};
      } else if constexpr (dim == 3) {
        using T = decltype(Auto(accessor(0, 0, 0)));
        return cuda::std::array<T, 8>{accessor(0, 0, 0), accessor(1, 0, 0), accessor(1, 1, 0), accessor(0, 1, 0), //
                                      accessor(0, 0, 1), accessor(1, 0, 1), accessor(1, 1, 1), accessor(0, 1, 1)};
      }
    };
    cuda::std::array positions = rearrangeAndLinearize(std::forward<TAccessorPositions>(accessorPositions));
    cuda::std::array values = rearrangeAndLinearize(std::forward<TAccessorValues>(accessorValues));
    static_assert(std::is_same_v<decltype(positions), cuda::std::array<Vec<Real, dim>, pow<dim>(2)>>,
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
      cuda::std::array<Vec<Real, dim>, dim> primitiveVertices;
      ForEach<dim>([&](auto i) {
        const int8_t *vert = marching_cube::detail::MarchingCubesEdges<dim>()[edge[i]];
        const Vec<Real, dim> p0 = positions[vert[0]];
        const Vec<Real, dim> p1 = positions[vert[1]];
        const Real v0 = values[vert[0]];
        const Real v1 = values[vert[1]];
        const float t = (isoValue - v0) / (v1 - v0);
        primitiveVertices[i] = Lerp(p0, p1, t);
      });

      bool success = true;
      ForEach<dim>([&](auto i) {
        constexpr int j = (i + 1) % dim;
        if (primitiveVertices[i] == primitiveVertices[j])
          success = false;
      });
      if (!success)
        continue;

      f(primitiveVertices);
    }
  }
};

} // namespace ARIA
