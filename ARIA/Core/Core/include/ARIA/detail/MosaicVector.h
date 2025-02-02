#pragma once

#include "ARIA/detail/MosaicIterator.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace ARIA {

namespace mosaic::detail {

template <MosaicPattern TMosaicPattern, typename TSpaceHostOrDevice>
struct reduce_mosaic_vector_storage_type;

template <MosaicPattern TMosaicPattern>
struct reduce_mosaic_vector_storage_type<TMosaicPattern, SpaceHost> {
private:
  template <type_array::detail::NonArrayType... Ts>
  consteval auto impl(MakeTypeArray<Ts...>) {
    using TStorage = Tup<thrust::host_vector<Ts...>>;
    return TStorage{};
  }

public:
  using type = decltype(impl(std::declval<mosaic_pattern_types_recursive_t<TMosaicPattern>>()));
};

template <MosaicPattern TMosaicPattern>
struct reduce_mosaic_vector_storage_type<TMosaicPattern, SpaceDevice> {
private:
  template <type_array::detail::NonArrayType... Ts>
  consteval auto impl(MakeTypeArray<Ts...>) {
    using TStorage = Tup<thrust::device_vector<Ts...>>;
    return TStorage{};
  }

public:
  using type = decltype(impl(std::declval<mosaic_pattern_types_recursive_t<TMosaicPattern>>()));
};

template <MosaicPattern TMosaicPattern, typename TSpaceHostOrDevice>
using reduce_mosaic_vector_storage_type_t =
    typename reduce_mosaic_vector_storage_type<TMosaicPattern, TSpaceHostOrDevice>::type;

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
  using TStorage = reduce_mosaic_vector_storage_type_t<TMosaicPattern, SpaceHost>;

  static constexpr size_t size = tuple_size_recursive_v<TMosaicPattern>;

public:
  using value_type = T;

private:
  TStorage storage_;
};

} // namespace mosaic::detail

} // namespace ARIA
