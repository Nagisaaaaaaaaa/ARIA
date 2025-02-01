#pragma once

#include "ARIA/Mosaic.h"
#include "ARIA/Property.h"
#include "ARIA/Tup.h"

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>

namespace ARIA {

template <typename TMosaic, typename TReferences>
  requires(mosaic::detail::is_mosaic_v<TMosaic>)
class MosaicReference final : public property::detail::PropertyBase<MosaicReference<TMosaic, TReferences>> {
private:
  static_assert(mosaic::detail::ValidMosaic<TMosaic>, "The mosaic definition is invalid");

  using T = typename mosaic::detail::is_mosaic<TMosaic>::T;
  using TMosaicPattern = typename mosaic::detail::is_mosaic<TMosaic>::TMosaicPattern;

  static constexpr size_t size = mosaic::detail::tuple_size_recursive_v<TMosaicPattern>;
  static_assert(size == boost::tuples::length<TReferences>::value,
                "The iterator types are inconsistent with the mosaic pattern");

public:
  ARIA_HOST_DEVICE constexpr explicit MosaicReference(const TReferences &references) : references_(references) {
    ForEach<size>([&]<auto i>() {
      static_assert(
          std::is_same_v<std::decay_t<decltype(Auto(mosaic::detail::get_recursive<i>(std::declval<TMosaicPattern>())))>,
                         std::decay_t<decltype(Auto(references_.template get<i>()))>>,
          "The iterator types are inconsistent with the mosaic pattern");
    });
  }

  ARIA_COPY_MOVE_ABILITY(MosaicReference, default, default);

public:
  ARIA_HOST_DEVICE constexpr T value() const {
    TMosaicPattern mosaicPattern;
    ForEach<size>([&]<auto i>() { mosaic::detail::get_recursive<i>(mosaicPattern) = references_.template get<i>(); });
    return TMosaic{}(mosaicPattern);
  }

  ARIA_HOST_DEVICE constexpr operator T() const { return value(); }

  template <typename U>
  ARIA_HOST_DEVICE constexpr MosaicReference &operator=(U &&arg) {
    T value = std::forward<U>(arg);
    TMosaicPattern mosaicPattern = TMosaic{}(value);
    ForEach<size>([&]<auto i>() { references_.template get<i>() = mosaic::detail::get_recursive<i>(mosaicPattern); });
    return *this;
  }

  template <typename U, size_t n>
  ARIA_HOST_DEVICE constexpr MosaicReference &operator=(const U (&args)[n]) {
    operator=(property::detail::ConstructWithArray<T>(args, std::make_index_sequence<n>{}));
    return *this;
  }

private:
  TReferences references_;
};

//
//
//
template <typename TMosaic, typename... TIterators>
  requires(mosaic::detail::is_mosaic_v<TMosaic>)
ARIA_HOST_DEVICE static constexpr auto make_mosaic_iterator(const Tup<TIterators...> &iterators) {
  boost::tuple<TIterators...> iteratorsBoost;
  ForEach<sizeof...(TIterators)>([&]<auto i>() { get<i>(iteratorsBoost) = get<i>(iterators); });

  return boost::make_transform_iterator(boost::make_zip_iterator(iteratorsBoost),
                                        []<typename TReferences>(const TReferences &references) {
    return MosaicReference<TMosaic, TReferences>{references};
  });
}

template <typename TNonMosaic, typename TIterator>
  requires(!mosaic::detail::is_mosaic_v<TNonMosaic>)
ARIA_HOST_DEVICE static constexpr auto make_mosaic_iterator(const Tup<TIterator> &iterator) {
  return get<0>(iterator);
}

} // namespace ARIA
