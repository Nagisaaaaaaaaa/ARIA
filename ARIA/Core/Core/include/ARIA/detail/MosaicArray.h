#pragma once

#include "ARIA/detail/MosaicIterator.h"

#include <cuda/std/array>

namespace ARIA {

namespace mosaic::detail {

// Given the `MosaicPattern` and `size`,
// deduce the storage type of the `MosaicArray`.
// It will be something such as `Tup<cuda::std::array<...>, ...>`.
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
// \brief Just like `std::array` and `cuda::std::array`, but
// only accept `Mosaic`s as the first template parameter.
//
// The internal storages are structure of arrays (SoA) which are consistent with
// the definitions of `MosaicPattern`s.
// Accessors such as `operator[]` and `begin` are implemented similar to `std::vector<bool>`.
template <typename TMosaic_, size_t size_>
  requires(is_mosaic_v<TMosaic_>)
class MosaicArray final {
public:
  using TMosaic = TMosaic_;

  [[nodiscard]] ARIA_HOST_DEVICE static consteval size_t size() { return size_; }

private:
  static_assert(ValidMosaic<TMosaic>, "The mosaic definition is invalid");

  using T = typename is_mosaic<TMosaic>::T;
  using TMosaicPattern = typename is_mosaic<TMosaic>::TMosaicPattern;
  using TStorage = mosaic_array_storage_type_t<TMosaicPattern, size()>;

public:
  using value_type = T;

public:
  constexpr MosaicArray() = default;

#if 0
  ARIA_HOST_DEVICE constexpr MosaicArray(const std::array<T, size()> &v) { operator=(v); }
#endif

  ARIA_HOST_DEVICE constexpr MosaicArray(const cuda::std::array<T, size()> &v) { operator=(v); }

#if 0
  ARIA_HOST_DEVICE constexpr MosaicArray &operator=(const std::array<T, size()> &v) {
    ForEach<size()>([&]<auto i>() { operator[](i) = v[i]; });
    return *this;
  }
#endif

  ARIA_HOST_DEVICE constexpr MosaicArray &operator=(const cuda::std::array<T, size()> &v) {
    ForEach<size()>([&]<auto i>() { operator[](i) = v[i]; });
    return *this;
  }

  ARIA_COPY_MOVE_ABILITY(MosaicArray, default, default);

public:
  [[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator[](size_t i) const { return *(data() + i); }

  [[nodiscard]] ARIA_HOST_DEVICE constexpr auto operator[](size_t i) { return *(data() + i); }

public:
  ARIA_HOST_DEVICE constexpr void fill(const T &value) {
    ForEach<size()>([&]<auto i>() { operator[](i) = value; });
  }

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

//! Unable to define CTAD for `MosaicArray` because
//! `TMosaic` can never be deduced by the constructor parameters.

//
//
//
// When `T` is `Mosaic`, reduce `Array` to `MosaicArray`,
// else, reduce to `cuda::std::array`.
template <typename T, size_t size>
struct reduce_array;

template <typename TMosaic, size_t size>
  requires(is_mosaic_v<TMosaic>)
struct reduce_array<TMosaic, size> {
  using type = MosaicArray<TMosaic, size>;
};

template <typename T, size_t size>
  requires(!is_mosaic_v<T>)
struct reduce_array<T, size> {
  using type = cuda::std::array<T, size>;
};

template <typename T, size_t size>
using reduce_array_t = typename reduce_array<T, size>::type;

} // namespace mosaic::detail

} // namespace ARIA
