#pragma once

#include "ARIA/detail/MosaicIterator.h"

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

namespace ARIA {

namespace mosaic::detail {

// Given the `MosaicPattern` and some other template parameters,
// deduce the storage type of the `MosaicVector`.
// It will be something such as `Tup<thrust::host_vector<...>, thrust::host_vector<...>, ...>`.
template <MosaicPattern TMosaicPattern, typename TSpaceHostOrDevice, typename... Ts>
struct mosaic_vector_storage_type;

template <MosaicPattern TMosaicPattern, typename... Ts>
struct mosaic_vector_storage_type<TMosaicPattern, SpaceHost, Ts...> {
private:
  template <type_array::detail::NonArrayType... Us>
  static consteval auto impl(TypeArray<Us...>) {
    // Use `thrust::host_vector` for `SpaceHost`.
    using TStorage = Tup<thrust::host_vector<Us, Ts...>...>;
    return TStorage{};
  }

public:
  using type = decltype(impl(std::declval<mosaic_pattern_types_recursive_t<TMosaicPattern>>()));
};

template <MosaicPattern TMosaicPattern, typename... Ts>
struct mosaic_vector_storage_type<TMosaicPattern, SpaceDevice, Ts...> {
private:
  template <type_array::detail::NonArrayType... Us>
  static consteval auto impl(TypeArray<Us...>) {
    // Use `thrust::device_vector` for `SpaceDevice`.
    using TStorage = Tup<thrust::device_vector<Us, Ts...>...>;
    return TStorage{};
  }

public:
  using type = decltype(impl(std::declval<mosaic_pattern_types_recursive_t<TMosaicPattern>>()));
};

template <MosaicPattern TMosaicPattern, typename TSpaceHostOrDevice, typename... Ts>
using mosaic_vector_storage_type_t =
    typename mosaic_vector_storage_type<TMosaicPattern, TSpaceHostOrDevice, Ts...>::type;

//
//
//
template <typename TMosaic_, typename TSpaceHostOrDevice, typename... Ts>
  requires(is_mosaic_v<TMosaic_>)
class MosaicVector final {
public:
  using TMosaic = TMosaic_;

private:
  static_assert(ValidMosaic<TMosaic>, "The mosaic definition is invalid");

  using T = typename is_mosaic<TMosaic>::T;
  using TMosaicPattern = typename is_mosaic<TMosaic>::TMosaicPattern;
  using TStorage = mosaic_vector_storage_type_t<TMosaicPattern, TSpaceHostOrDevice, Ts...>;

  template <typename UMosaic, typename USpaceHostOrDevice, typename... Us>
    requires(is_mosaic_v<UMosaic>)
  friend class MosaicVector;

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

  template <typename USpaceHostOrDevice, typename... Us>
    requires(!std::is_same_v<MosaicVector, MosaicVector<TMosaic, USpaceHostOrDevice, Us...>>)
  MosaicVector(const MosaicVector<TMosaic, USpaceHostOrDevice, Us...> &v) {
    operator=(v);
  }

  template <typename USpaceHostOrDevice, typename... Us>
    requires(!std::is_same_v<MosaicVector, MosaicVector<TMosaic, USpaceHostOrDevice, Us...>>)
  MosaicVector &operator=(const MosaicVector<TMosaic, USpaceHostOrDevice, Us...> &v) {
    ForEach<rank_v<TStorage>>([&]<auto i>() { get<i>(storage_) = get<i>(v.storage_); });
    return *this;
  }

  ARIA_COPY_MOVE_ABILITY(MosaicVector, default, default);

public:
  constexpr size_t size() const { return get<0>(storage_).size(); }

  constexpr auto operator[](size_t i) const { return *(data() + i); }

  constexpr auto operator[](size_t i) { return *(data() + i); }

  constexpr void resize(size_t n) {
    ForEach<rank_v<TStorage>>([&]<auto i>() { get<i>(storage_).resize(n); });
  }

  constexpr void clear() {
    ForEach<rank_v<TStorage>>([&]<auto i>() { get<i>(storage_).clear(); });
  }

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

//! CTAD is not supported because `TMosaic` can never
//! be deduced by the constructor parameters.

} // namespace mosaic::detail

} // namespace ARIA
