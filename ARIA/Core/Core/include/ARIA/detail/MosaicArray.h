#pragma once

#include "ARIA/detail/MosaicIterator.h"

#include <cuda/std/array>

namespace ARIA {

namespace mosaic::detail {

template <MosaicPattern TMosaicPattern, size_t size>
struct mosaic_array_storage_type {
private:
  template <type_array::detail::NonArrayType... Us>
  static consteval auto impl(TypeArray<Us...>) {
    // Use `cuda::std::array`.
    using TStorage = Tup<cuda::std::array<Us, size>...>;
    return TStorage{};
  }

public:
  using type = decltype(impl(std::declval<mosaic_pattern_types_recursive_t<TMosaicPattern>>()));
};

template <MosaicPattern TMosaicPattern, size_t size>
using mosaic_array_storage_type_t = typename mosaic_array_storage_type<TMosaicPattern, size>::type;

//
//
//
//
//

} // namespace mosaic::detail

} // namespace ARIA
