#pragma once

#include "ARIA/Mosaic.h"
#include "ARIA/Tup.h"

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>

namespace ARIA {

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
