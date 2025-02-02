#pragma once

#include "ARIA/detail/MosaicIterator.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace ARIA {

namespace mosaic::detail {

template <MosaicPattern TMosaicPattern, typename TSpaceHostOrDevice, typename... Ts>
struct reduce_mosaic_vector_storage_type;

template <MosaicPattern TMosaicPattern, typename... Ts>
struct reduce_mosaic_vector_storage_type<TMosaicPattern, SpaceHost, Ts...> {
private:
  template <type_array::detail::NonArrayType... Us>
  consteval auto impl(MakeTypeArray<Us...>) {
    using TStorage = Tup<thrust::host_vector<Us, Ts...>...>;
    return TStorage{};
  }

public:
  using type = decltype(impl(std::declval<mosaic_pattern_types_recursive_t<TMosaicPattern>>()));
};

template <MosaicPattern TMosaicPattern, typename... Ts>
struct reduce_mosaic_vector_storage_type<TMosaicPattern, SpaceDevice, Ts...> {
private:
  template <type_array::detail::NonArrayType... Us>
  consteval auto impl(MakeTypeArray<Us...>) {
    using TStorage = Tup<thrust::device_vector<Us, Ts...>...>;
    return TStorage{};
  }

public:
  using type = decltype(impl(std::declval<mosaic_pattern_types_recursive_t<TMosaicPattern>>()));
};

template <MosaicPattern TMosaicPattern, typename TSpaceHostOrDevice, typename... Ts>
using reduce_mosaic_vector_storage_type_t =
    typename reduce_mosaic_vector_storage_type<TMosaicPattern, TSpaceHostOrDevice, Ts...>::type;

//
//
//
template <typename TMosaic, typename... Ts>
  requires(is_mosaic_v<TMosaic>)
class MosaicVector final {
private:
  static_assert(ValidMosaic<TMosaic>, "The mosaic definition is invalid");

  using T = typename is_mosaic<TMosaic>::T;
  using TMosaicPattern = typename is_mosaic<TMosaic>::TMosaicPattern;
  using TStorage = reduce_mosaic_vector_storage_type_t<TMosaicPattern, SpaceHost, Ts...>;

  static constexpr size_t size = tuple_size_recursive_v<TMosaicPattern>;

public:
  using value_type = T;

private:
  TStorage storage_;
};

} // namespace mosaic::detail

} // namespace ARIA
