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
  constexpr MosaicVector() = default;

  constexpr explicit MosaicVector(size_t n) { resize(n); }

  constexpr MosaicVector(std::initializer_list<T> list) : MosaicVector(list.size()) {
    const T *begin = list.begin();
    for (size_t i = 0; i < list.size(); ++i)
      operator[](i) = *(begin + i);
  }

  ARIA_COPY_MOVE_ABILITY(MosaicVector, default, default);

public:
  constexpr auto operator[](size_t i) const { return *(data() + i); }

  constexpr auto operator[](size_t i) { return *(data() + i); }

  constexpr void resize(size_t n) {
    ForEach<rank_v<TStorage>>([&]<auto i>() { get<i>(storage_).resize(n); });
  }

  constexpr size_t size() const { return get<0>(storage_).size(); }

  constexpr auto begin() {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_iterator<TMosaic>(Tup{std::forward<S>(s).begin()...});
    });
  }

  constexpr auto end() {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_iterator<TMosaic>(Tup{std::forward<S>(s).end()...});
    });
  }

  constexpr auto cbegin() const {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_iterator<TMosaic>(Tup{std::forward<S>(s).cbegin()...});
    });
  }

  constexpr auto cend() const {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_iterator<TMosaic>(Tup{std::forward<S>(s).cend()...});
    });
  }

  constexpr auto data() const {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_pointer<TMosaic>(Tup{std::forward<S>(s).data()...});
    });
  }

  constexpr auto data() {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_pointer<TMosaic>(Tup{std::forward<S>(s).data()...});
    });
  }

private:
  TStorage storage_;
};

} // namespace mosaic::detail

} // namespace ARIA
