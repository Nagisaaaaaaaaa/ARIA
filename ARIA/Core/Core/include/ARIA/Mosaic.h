#pragma once

// TODO: Mosaic is an abstraction about:
//       How to describe instances of one type with instances of another type.
//       Eg 1. `double` can be described with `float` with precision lost.
//       Eg 2. `double` can be described with `double` itself.
//       Eg 3. `Vec3f` can be described with `struct { float, float, float }`.

#include "ARIA/Property.h"

#include <boost/pfr.hpp>

namespace ARIA {

template <typename T>
[[nodiscard]] static consteval bool IsMosaicPatternImpl() {
  static_assert(std::is_same_v<T, std::decay_t<T>>, "The given type `T` should be a decayed type");

  // For scalar types.
  if constexpr (std::is_scalar_v<T>)
    return true;
  // For aggregate types.
  //! `> 1` is required here because classes with only one member are considered unnecessary.
  else if constexpr (std::is_aggregate_v<T> && boost::pfr::tuple_size_v<T> > 1) {
    bool res = true;
    ForEach<boost::pfr::tuple_size_v<T>>([&](auto i) {
      using U = std::decay_t<decltype(boost::pfr::get<i>(std::declval<T>()))>;
      if (!IsMosaicPatternImpl<U>())
        res = false;
    });
    return res;
  }

  return false;
}

template <typename T>
static constexpr bool is_mosaic_pattern_v = IsMosaicPatternImpl<T>();

template <typename T>
concept MosaicPattern = is_mosaic_pattern_v<T>;

//
//
//
template <typename T, MosaicPattern U>
class Mosaic {};

} // namespace ARIA
