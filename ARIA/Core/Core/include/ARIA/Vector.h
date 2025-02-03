#pragma once

#include "ARIA/detail/MosaicVector.h"

namespace ARIA {

template <typename T, typename TSpaceHostOrDevice, typename... Ts>
using Vector = mosaic::detail::reduce_vector_t<T, TSpaceHostOrDevice, Ts...>;

template <typename T, typename... Ts>
using VectorHost = Vector<T, SpaceHost, Ts...>;

template <typename T, typename... Ts>
using VectorDevice = Vector<T, SpaceDevice, Ts...>;

} // namespace ARIA
