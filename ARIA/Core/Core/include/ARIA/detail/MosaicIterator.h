#pragma once

#include "ARIA/Mosaic.h"

#include <boost/iterator/transform_iterator.hpp>
#include <boost/iterator/zip_iterator.hpp>

namespace ARIA {

template <typename TMosaic, typename... TIterators>
  requires(mosaic::detail::is_mosaic_v<TMosaic>)
ARIA_HOST_DEVICE static constexpr auto make_mosaic_iterator(TIterators &&...iterators) {
  static_assert(mosaic::detail::ValidMosaic<TMosaic>, "The mosaic definition is invalid");
  using TMosaicPattern = typename mosaic::detail::is_mosaic<TMosaic>::TMosaicPattern;

  return boost::make_transform_iterator(
      boost::make_zip_iterator(boost::make_tuple(std::forward<TIterators>(iterators)...)),
      [](const auto &z) -> TMosaicPattern {
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

} // namespace ARIA
