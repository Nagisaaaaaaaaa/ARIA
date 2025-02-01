#pragma once

#include "ARIA/Mosaic.h"
#include "ARIA/Property.h"
#include "ARIA/Tup.h"

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>

namespace ARIA {

template <typename TMosaic, typename TReference>
  requires(mosaic::detail::is_mosaic_v<TMosaic>)
class MosaicReference final : public property::detail::PropertyBase<MosaicReference<TMosaic, TReference>> {
private:
  static_assert(mosaic::detail::ValidMosaic<TMosaic>, "The mosaic definition is invalid");

  using T = typename mosaic::detail::is_mosaic<TMosaic>::T;
  using TMosaicPattern = typename mosaic::detail::is_mosaic<TMosaic>::TMosaicPattern;

  static constexpr size_t size = boost::tuples::length<TReference>::value;
  static_assert(size == mosaic::detail::tuple_size_recursive_v<TMosaicPattern>,
                "The iterator types are inconsistent with the mosaic pattern");

public:
  ARIA_HOST_DEVICE constexpr explicit MosaicReference(TReference reference) : reference_(reference) {}

  ARIA_COPY_MOVE_ABILITY(MosaicReference, default, default);

  ARIA_HOST_DEVICE constexpr T value() const {
    TMosaicPattern mosaicPattern;
    ForEach<size>([&]<auto i>() {
      static_assert(std::is_same_v<std::decay_t<decltype(mosaic::detail::get_recursive<i>(mosaicPattern))>,
                                   std::decay_t<decltype(reference_.template get<i>())>>,
                    "The iterator types are inconsistent with the mosaic pattern");
      mosaic::detail::get_recursive<i>(mosaicPattern) = reference_.template get<i>();
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
                                   std::decay_t<decltype(reference_.template get<i>())>>,
                    "The iterator types are inconsistent with the mosaic pattern");
      reference_.template get<i>() = mosaic::detail::get_recursive<i>(mosaicPattern);
    });
    return *this;
  }

private:
  TReference reference_;
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

  return boost::make_transform_iterator(boost::make_zip_iterator(iteratorsBoost), [](const auto &z) -> TMosaicPattern {
    TMosaicPattern res;
    ForEach<sizeof...(TIterators)>([&]<auto i>() {
      static_assert(std::is_same_v<std::decay_t<decltype(mosaic::detail::get_recursive<i>(res))>,
                                   std::decay_t<decltype(z.template get<i>())>>,
                    "The iterator types are inconsistent with the mosaic pattern");
      mosaic::detail::get_recursive<i>(res) = z.template get<i>();
    });
    return res;
  });
}

template <typename TNonMosaic, typename TIterator>
  requires(!mosaic::detail::is_mosaic_v<TNonMosaic>)
ARIA_HOST_DEVICE static constexpr auto make_mosaic_iterator(TIterator iterator) {
  return iterator;
}

} // namespace ARIA
