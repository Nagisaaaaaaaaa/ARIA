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
template <typename TMosaic_, size_t size>
  requires(is_mosaic_v<TMosaic_>)
class MosaicArray final {
public:
  using TMosaic = TMosaic_;

private:
  static_assert(ValidMosaic<TMosaic>, "The mosaic definition is invalid");

  using T = typename is_mosaic<TMosaic>::T;
  using TMosaicPattern = typename is_mosaic<TMosaic>::TMosaicPattern;
  using TStorage = mosaic_array_storage_type_t<TMosaicPattern, size>;

  // `friend` is added to access `storage_` of other `MosaicArray` types.
  template <typename UMosaic, size_t size1>
    requires(is_mosaic_v<UMosaic>)
  friend class MosaicArray;

public:
  using value_type = T;

public:
  constexpr MosaicArray() = default;

  ARIA_HOST_DEVICE constexpr MosaicArray(const std::array<T, size> &v) { operator=(v); }

  ARIA_HOST_DEVICE MosaicArray &operator=(const std::array<T, size> &v) {
    ForEach<size>([&]<auto i>() { operator[](i) = v[i]; });
    return *this;
  }

  ARIA_COPY_MOVE_ABILITY(MosaicArray, default, default);

public:
  [[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator[](size_t i) const { return *(data() + i); }

  [[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator[](size_t i) { return *(data() + i); }

public:
  [[nodiscard]] ARIA_HOST_DEVICE constexpr auto begin() {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_iterator<TMosaic>(Tup{std::forward<S>(s).begin()...});
    });
  }

  [[nodiscard]] ARIA_HOST_DEVICE constexpr auto end() {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_iterator<TMosaic>(Tup{std::forward<S>(s).end()...});
    });
  }

  [[nodiscard]] ARIA_HOST_DEVICE constexpr auto cbegin() const {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_iterator<TMosaic>(Tup{std::forward<S>(s).cbegin()...});
    });
  }

  [[nodiscard]] ARIA_HOST_DEVICE constexpr auto cend() const {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_iterator<TMosaic>(Tup{std::forward<S>(s).cend()...});
    });
  }

  [[nodiscard]] ARIA_HOST_DEVICE constexpr auto data() const {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_pointer<TMosaic>(Tup{std::forward<S>(s).data()...});
    });
  }

  [[nodiscard]] ARIA_HOST_DEVICE constexpr auto data() {
    return cute::apply(storage_, []<typename... S>(S &&...s) {
      return make_mosaic_pointer<TMosaic>(Tup{std::forward<S>(s).data()...});
    });
  }

private:
  TStorage storage_;
};

} // namespace mosaic::detail

} // namespace ARIA
