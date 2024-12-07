#pragma once

// TODO: Mosaic is an abstraction about:
//       How to describe instances of one type with instances of another type.
//       Eg 1. `double` can be described with `float` with precision lost.
//       Eg 2. `double` can be described with `double` itself.
//       Eg 3. `Vec3f` can be described with `struct { float, float, float }`.

#include "ARIA/ForEach.h"
#include "ARIA/TypeArray.h"

#include <boost/pfr.hpp>

namespace ARIA {

template <typename T>
[[nodiscard]] static consteval bool IsMosaicPatternImpl() {
  static_assert(std::is_same_v<T, std::decay_t<T>>, "The given type should be a decayed type");

  //! The "strongest" copy and move ability is required, which means that
  //! types such as l-value or r-value references are not allowed here.
  if (!(std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T> && std::is_move_constructible_v<T> &&
        std::is_move_assignable_v<T>))
    return false;

  // For scalar types.
  if constexpr (std::is_scalar_v<T>)
    return true;
  // For aggregate types.
  //! `> 1` is required here because classes with only one member are considered unnecessary.
  else if constexpr (std::is_aggregate_v<T> && boost::pfr::tuple_size_v<T> > 1) {
    bool res = true;
    ForEach<boost::pfr::tuple_size_v<T>>([&](auto i) {
      using U = decltype(boost::pfr::get<i>(std::declval<T>()));
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
  else {
    TInteger sum = 0;
    ForEach<boost::pfr::tuple_size_v<T>>([&](auto i) {
      using U = decltype(boost::pfr::get<i>(std::declval<T>()));
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
// \brief Given the recursive index, compute the non-recursive index.
//
// \example ```cpp
// struct Pattern {
//   int v0;
//
//   struct {
//     int v1;
//     int v2;
//   } s0;
// };
//
// static_assert(IRec2INonRec<0, Pattern>() == 0);
// static_assert(IRec2INonRec<1, Pattern>() == 1);
// static_assert(IRec2INonRec<2, Pattern>() == 1);
// static_assert(IRec2INonRec<3, Pattern>() == 2);
// static_assert(IRec2INonRec<99999, Pattern>() == 2);
// ```
template <auto iRec, MosaicPattern T>
[[nodiscard]] static consteval auto IRec2INonRec() {
  using TInteger = std::decay_t<decltype(boost::pfr::tuple_size_v<T>)>;

  TInteger sum = 0;
  TInteger iNonRec = 0;

  ForEach<boost::pfr::tuple_size_v<T>>([&](auto i) {
    using U = decltype(boost::pfr::get<i>(std::declval<T>()));
    sum += tuple_size_recursive_v<U>;

    if (iRec >= sum)
      iNonRec = i + 1;
  });

  return iNonRec;
}

//
//
//
// \brief Given the non-recursive index, compute the recursive index.
//
// \example ```cpp
// struct Pattern {
//   int v0;
//
//   struct {
//     int v1;
//     int v2;
//   } s0;
// };
//
// static_assert(INonRec2IRec<0, Pattern>() == 0);
// static_assert(INonRec2IRec<1, Pattern>() == 1);
// static_assert(INonRec2IRec<2, Pattern>() == 3);
// static_assert(INonRec2IRec<99999, Pattern>() == 3);
template <auto iNonRec, MosaicPattern T>
[[nodiscard]] static consteval auto INonRec2IRec() {
  using TInteger = std::decay_t<decltype(boost::pfr::tuple_size_v<T>)>;

  TInteger sum = 0;

  ForEach<boost::pfr::tuple_size_v<T>>([&](auto i) {
    if (i >= iNonRec)
      return;

    using U = decltype(boost::pfr::get<i>(std::declval<T>()));
    sum += tuple_size_recursive_v<U>;
  });

  return sum;
}

//
//
//
template <auto iRec, typename T>
[[nodiscard]] static inline constexpr decltype(auto) get_recursive(T &&v) noexcept {
  using TDecayed = std::decay_t<T>;
  static_assert(MosaicPattern<TDecayed>, "The decayed given type should be a `MosaicPattern`");
  static_assert(iRec < tuple_size_recursive_v<TDecayed>, "Index out of range");

  using TInteger = std::decay_t<decltype(boost::pfr::tuple_size_v<TDecayed>)>;

  if constexpr (std::is_scalar_v<TDecayed>)
    return boost::pfr::get<iRec>(std::forward<T>(v));
  else {
    constexpr TInteger iNonRec = IRec2INonRec<iRec, TDecayed>();
    return get_recursive<iRec - INonRec2IRec<iNonRec, TDecayed>()>(boost::pfr::get<iNonRec>(std::forward<T>(v)));
  }
}

//
//
//
namespace mosaic::detail {

template <MosaicPattern T, auto i, typename TArray>
struct MosaicTilesImpl;

template <MosaicPattern T, auto i, typename TArray>
  requires(i < tuple_size_recursive_v<T>)
struct MosaicTilesImpl<T, i, TArray> {
  using type =
      typename MosaicTilesImpl<T, i + 1, MakeTypeArray<TArray, decltype(get_recursive<i>(std::declval<T>()))>>::type;
};

template <MosaicPattern T, auto i, typename TArray>
  requires(i == tuple_size_recursive_v<T>)
struct MosaicTilesImpl<T, i, TArray> {
  using type = TArray;
};

} // namespace mosaic::detail

template <MosaicPattern T>
using MosaicTiles = typename mosaic::detail::MosaicTilesImpl<T, 0, MakeTypeArray<>>::type;

//
//
//
template <typename T, MosaicPattern U>
class Mosaic;

//
//
//
template <typename T>
struct is_mosaic : std::false_type {};

template <typename T_, MosaicPattern U_>
struct is_mosaic<Mosaic<T_, U_>> : std::true_type {
  using T = T_;
  using U = U_;
};

template <typename T>
static constexpr bool is_mosaic_v = is_mosaic<T>::value;

//
//
//
template <typename TMosaic>
  requires(is_mosaic_v<TMosaic>)
[[nodiscard]] static consteval bool IsValidMosaicImpl() {
  using T = typename is_mosaic<TMosaic>::T;
  using U = typename is_mosaic<TMosaic>::U;

  if constexpr (!(std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<T>())), U> &&
                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<U>())), T> &&

                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<T &>())), U> &&
                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<U &>())), T> &&

                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<T &&>())), U> &&
                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<U &&>())), T> &&

                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const T>())), U> &&
                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const U>())), T> &&

                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const T &>())), U> &&
                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const U &>())), T> &&

                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const T &&>())), U> &&
                  std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const U &&>())), T>))
    return false;

  return true;
}

template <typename T>
static constexpr bool is_valid_mosaic_v = IsValidMosaicImpl<T>();

template <typename T>
concept ValidMosaic = is_valid_mosaic_v<T>;

} // namespace ARIA
