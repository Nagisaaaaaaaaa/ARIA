#pragma once

namespace ARIA {

namespace mosaic::detail {

// `MosaicPattern`s are classes which can be easily serialized.
// For example:
// 1. `int` and `int *` are "scalar" types, which
//    can be serialized by themselves.
// 2. `struct { int x, y; }` and `struct { int x; struct { int y, z; } s; }` are "aggregate" types, where
//    all members are recursively "scalar" types.
//! 3. `std::string` and `std::vector` are "non-scalar-and-non-aggregate" types.
//!    They are considered complex, thus cannot be easily serialized.
//
// It is named as "pattern" because, you can imagine that,
// small classes can be arbitrary placed together and be merged into a large class.
// For example, `Vec3f` = 3 `float`s = `struct { float x, y; }` + `float`.
//
//! This kind of serialization may be different from others.
//! Here are the main features:
//! 1. Compile-time number of elements (tuple size):
//!    `Vec3f` may be split into 3 `float`s at compile time, and
//!    it is unable to serialize `std::string`.
//! 2. Named elements:
//!    `MosaicPattern`s are defined by structures, not tuples, so
//!    every elements are required to be named.
//! 3. Any definition is allowed, as long as you can recover the type:
//!    `double` can be serialized with `float`, but precision lost.
//!    `float` can be serialized with `double`, but nothing better.
//!    `float` can be serialized with `float`, of course.
template <typename T>
[[nodiscard]] static consteval bool IsMosaicPatternImpl() {
  static_assert(std::is_same_v<T, std::decay_t<T>>, "The given type should be a decayed type");

  //! The "strongest" copy and move ability is required, which means that
  //! types such as l-value or r-value references are not allowed here.
  if (!(std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T> && std::is_move_constructible_v<T> &&
        std::is_move_assignable_v<T>))
    return false;

  // For aggregate types.
  //! `> 1` is required here because classes with only one member are considered unnecessary.
  if constexpr (std::is_aggregate_v<T>) {
    if (boost::pfr::tuple_size_v<T> <= 1)
      return false;

    bool res = true;
    ForEach<boost::pfr::tuple_size_v<T>>([&](auto i) {
      using U = decltype(boost::pfr::get<i>(std::declval<T>()));
      if (!IsMosaicPatternImpl<U>())
        res = false;
    });
    return res;
  }
  // For non-aggregate types (scalar types, or non-scalar-and-non-aggregate types).
  else {
    static_assert(std::is_scalar_v<T>, "Non-scalar-and-non-aggregate types such as `std::string` cannot be "
                                       "perfectly handled by `boost::pfr`, so these types are strictly forbidden.");

    return true;
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
// As we have known, one of the mean features of `MosaicPattern` is
// compile-time number of elements (tuple size).
// It should be computed non-recursively or recursively.
template <MosaicPattern T>
[[nodiscard]] static consteval auto TupleSizeRecursiveImpl() {
  using TInteger = std::decay_t<decltype(boost::pfr::tuple_size_v<T>)>;

  if constexpr (std::is_aggregate_v<T>) {
    TInteger sum = 0;
    ForEach<boost::pfr::tuple_size_v<T>>([&](auto i) {
      using U = decltype(boost::pfr::get<i>(std::declval<T>()));
      sum += TupleSizeRecursiveImpl<U>();
    });
    return sum;
  } else
    return TInteger{1};
}

template <MosaicPattern T>
static constexpr auto tuple_size_recursive_v = TupleSizeRecursiveImpl<T>();

//
//
//
// Before continue, we need to define some supporting functions.

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
// We should be able to recursively or non-recursively
// iterate through all the elements of a `MosaicPattern` instance.
template <auto iRec, typename T>
[[nodiscard]] static inline constexpr decltype(auto) get_recursive(T &&v) noexcept {
  using TDecayed = std::decay_t<T>;
  static_assert(MosaicPattern<TDecayed>, "The decayed given type should be a `MosaicPattern`");
  static_assert(iRec < tuple_size_recursive_v<TDecayed>, "Index out of range");

  using TInteger = std::decay_t<decltype(boost::pfr::tuple_size_v<TDecayed>)>;

  if constexpr (std::is_aggregate_v<TDecayed>) {
    constexpr TInteger iNonRec = IRec2INonRec<iRec, TDecayed>();
    return get_recursive<iRec - INonRec2IRec<iNonRec, TDecayed>()>(boost::pfr::get<iNonRec>(std::forward<T>(v)));
  } else
    return boost::pfr::get<iRec>(std::forward<T>(v));
}

//
//
//
// We can recursively gather all the types of a `MosaicPattern` into a `TypeArray`.
template <MosaicPattern T, auto i, typename TArray>
struct mosaic_pattern_types_recursive_impl;

template <MosaicPattern T, auto i, typename TArray>
  requires(i < tuple_size_recursive_v<T>)
struct mosaic_pattern_types_recursive_impl<T, i, TArray> {
  using type = typename mosaic_pattern_types_recursive_impl<
      T,
      i + 1,
      MakeTypeArray<TArray, decltype(get_recursive<i>(std::declval<T>()))>>::type;
};

template <MosaicPattern T, auto i, typename TArray>
  requires(i == tuple_size_recursive_v<T>)
struct mosaic_pattern_types_recursive_impl<T, i, TArray> {
  using type = TArray;
};

template <MosaicPattern T>
using mosaic_pattern_types_recursive_t = typename mosaic_pattern_types_recursive_impl<T, 0, MakeTypeArray<>>::type;

//
//
//
//
//
// Whether the given type is `Mosaic<T, TMosaicPattern>`.
// Here, the second template parameter is restricted with concept.
template <typename T>
struct is_mosaic : std::false_type {};

template <typename T_, MosaicPattern TMosaicPattern_>
struct is_mosaic<Mosaic<T_, TMosaicPattern_>> : std::true_type {
  using T = T_;
  using TMosaicPattern = TMosaicPattern_;
};

template <typename T>
static constexpr bool is_mosaic_v = is_mosaic<T>::value;

//
//
//
// Whether the given type is a valid `Mosaic`, where "valid" means that:
// 1. Types should be carefully checked.
// 2. `operator()` should be carefully checked.
template <typename TMosaic>
  requires(is_mosaic_v<TMosaic>)
[[nodiscard]] static consteval bool IsValidMosaicImpl() {
  using T = typename is_mosaic<TMosaic>::T;
  using P = typename is_mosaic<TMosaic>::TMosaicPattern;

  static_assert(std::is_same_v<T, std::decay_t<T>>, "The given type should be a decayed type");

  static_assert(std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T> && std::is_move_constructible_v<T> &&
                    std::is_move_assignable_v<T>,
                "The \"strongest\" copy and move ability is required");

  static_assert(std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<T>())), P> &&
                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<P>())), T> &&

                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<T &>())), P> &&
                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<P &>())), T> &&

                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<T &&>())), P> &&
                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<P &&>())), T> &&

                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const T>())), P> &&
                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const P>())), T> &&

                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const T &>())), P> &&
                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const P &>())), T> &&

                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const T &&>())), P> &&
                    std::is_same_v<decltype(std::declval<const TMosaic>()(std::declval<const P &&>())), T>,
                "`operator()` should be overloaded to support conversions to and from the mosaic pattern");

  return true;
}

template <typename T>
static constexpr bool is_valid_mosaic_v = IsValidMosaicImpl<T>();

template <typename T>
concept ValidMosaic = is_valid_mosaic_v<T>;

} // namespace mosaic::detail

} // namespace ARIA
