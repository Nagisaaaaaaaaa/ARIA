#pragma once

// TODO: Mosaic is an abstraction about:
//       How to describe instances of one type with instances of another type.
//       Eg 1. `double` can be described with `float` with precision lost.
//       Eg 2. `double` can be described with `double` itself.
//       Eg 3. `Vec3f` can be described with `struct { float, float, float }`.

#include "ARIA/ForEach.h"

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
template <MosaicPattern T>
[[nodiscard]] static consteval auto TupleSizeRecursiveImpl() {
  using TInteger = std::decay_t<decltype(boost::pfr::tuple_size_v<T>)>;

  if constexpr (std::is_scalar_v<T>)
    return TInteger{1};
  else if constexpr (std::is_aggregate_v<T>) {
    TInteger sum = 0;
    ForEach<boost::pfr::tuple_size_v<T>>([&](auto i) {
      using U = std::decay_t<decltype(boost::pfr::get<i>(std::declval<T>()))>;
      sum += TupleSizeRecursiveImpl<U>();
    });
    return sum;
  }
}

template <MosaicPattern T>
static constexpr auto tuple_size_recursive_v = TupleSizeRecursiveImpl<T>();

//
//
//
// Given the recursive index, compute the non-recursive index.
template <auto iRec, MosaicPattern T>
[[nodiscard]] static consteval auto IRec2INonRec() {
  using TInteger = std::decay_t<decltype(boost::pfr::tuple_size_v<T>)>;

  TInteger sum = 0;
  TInteger iNonRec = 0;

  ForEach<boost::pfr::tuple_size_v<T>>([&](auto i) {
    using U = std::decay_t<decltype(boost::pfr::get<i>(std::declval<T>()))>;
    sum += tuple_size_recursive_v<U>;

    if (iRec >= sum)
      iNonRec = i + 1;
  });

  return iNonRec;
}

//
//
//
template <typename T, MosaicPattern U>
class Mosaic {};

} // namespace ARIA
