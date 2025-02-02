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
  static consteval auto impl(TypeArray<Us...>) {
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
  static consteval auto impl(TypeArray<Us...>) {
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

public:
  using value_type = T;

public:
  constexpr MosaicVector() : MosaicVector(0) {}

  constexpr explicit MosaicVector(size_t n) { resize(n); }

  constexpr void resize(size_t n) {
    ForEach<rank_v<TStorage>>([&]<auto i>() { get<i>(storage_).resize(n); });
  }

  constexpr size_t size() const { return get<0>(storage_).size(); }

private:
  TStorage storage_;
};

} // namespace mosaic::detail

} // namespace ARIA
