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
  ARIA_HOST_DEVICE constexpr explicit MosaicReference(TReferences references) : references_(references) {}

  ARIA_COPY_MOVE_ABILITY(MosaicReference, default, default);

  ARIA_HOST_DEVICE constexpr T value() const {
    TMosaicPattern mosaicPattern;
    ForEach<size>([&]<auto i>() {
      static_assert(std::is_same_v<std::decay_t<decltype(mosaic::detail::get_recursive<i>(mosaicPattern))>,
                                   std::decay_t<decltype(references_.template get<i>())>>,
                    "The iterator types are inconsistent with the mosaic pattern");
      mosaic::detail::get_recursive<i>(mosaicPattern) = references_.template get<i>();
    });
    return TMosaic{}(mosaicPattern);
  }

  ARIA_HOST_DEVICE constexpr operator T() const { return value(); }

  template <typename U>
  ARIA_HOST_DEVICE constexpr MosaicReference &operator=(U &&value) {
    T v = std::forward<U>(value);
    TMosaicPattern mosaicPattern = TMosaic{}(v);
    ForEach<size>([&]<auto i>() {
      static_assert(std::is_same_v<std::decay_t<decltype(mosaic::detail::get_recursive<i>(mosaicPattern))>,
                                   std::decay_t<decltype(references_.template get<i>())>>,
                    "The iterator types are inconsistent with the mosaic pattern");
      references_.template get<i>() = mosaic::detail::get_recursive<i>(mosaicPattern);
    });
    return *this;
  }

  template <typename U, size_t n>
  ARIA_HOST_DEVICE constexpr MosaicReference &operator=(const U (&args)[n]) {
    return operator=(property::detail::ConstructWithArray<T>(args, std::make_index_sequence<n>{}));
  }

private:
  TReferences references_;
};

//
//
//
template <typename TMosaic, typename... TIterators>
  requires(mosaic::detail::is_mosaic_v<TMosaic>)
ARIA_HOST_DEVICE static constexpr auto make_mosaic_iterator(Tup<TIterators...> iterators) {
  static_assert(mosaic::detail::ValidMosaic<TMosaic>, "The mosaic definition is invalid");
  using TMosaicPattern = typename mosaic::detail::is_mosaic<TMosaic>::TMosaicPattern;

  boost::tuple<TIterators...> iteratorsBoost;
  ForEach<sizeof...(TIterators)>([&]<auto i>() { get<i>(iteratorsBoost) = get<i>(iterators); });

  return boost::make_transform_iterator(boost::make_zip_iterator(iteratorsBoost),
                                        []<typename Tz>(const Tz &z) { return MosaicReference<TMosaic, Tz>{z}; });
}

template <typename TNonMosaic, typename TIterator>
  requires(!mosaic::detail::is_mosaic_v<TNonMosaic>)
ARIA_HOST_DEVICE static constexpr auto make_mosaic_iterator(TIterator iterator) {
  return iterator;
}

} // namespace ARIA
