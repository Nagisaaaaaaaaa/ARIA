#pragma once

namespace ARIA {

template <typename T, auto d>
ARIA_HOST_DEVICE inline /*constexpr*/ AABB<T, d>::AABB()
    : infAndSup_{ConstructVecWithNEqualedValues(infinity<T>), ConstructVecWithNEqualedValues(-infinity<T>)} {}

} // namespace ARIA
