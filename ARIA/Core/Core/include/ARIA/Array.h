#pragma once

#include "ARIA/detail/MosaicArray.h"

namespace ARIA {

template <typename T, size_t size>
using Array = mosaic::detail::reduce_array_t<T, size>;

} // namespace ARIA
