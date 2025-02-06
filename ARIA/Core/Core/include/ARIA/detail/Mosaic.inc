#pragma once

namespace ARIA {

namespace mosaic::detail {

//! ARIA's `Mosaic` system is implemented mainly based on `boost::pfr`, but
//! unfortunately, it still has many limitations and undefined behaviors.
//! So, we have to extend the functionalities of `boost::pfr`, that is,
//! we will have wrappers for `boost::pfr::tuple_size` and `boost::pfr::get`.

// `array_trait` has the same `value` as `std::is_array` but
// contains more information about the array, including type and size.
template <typename T>
struct array_trait : std::false_type {};

template <typename T_, size_t n_>
struct array_trait<T_[n_]> : std::true_type {
  using T = T_;
  static constexpr size_t n = n_;
};

template <typename T_>
struct array_trait<T_[]> : std::true_type {
  using T = T_;
};

//
//
//
// A wrapper for `boost::pfr::tuple_size`, which takes C arrays into account.
// For example, `tuple_size_v<int[3]>` will be 3, not 1.
template <typename T>
struct tuple_size;

// For non-array types, simply call `boost::pfr::tuple_size_v`.
template <typename T>
  requires(!std::is_array_v<T>)
struct tuple_size<T> {
  static constexpr size_t value = boost::pfr::tuple_size_v<T>;
};

// For array types, use `array_trait`.
template <typename T>
  requires(std::is_array_v<T>)
struct tuple_size<T> {
  static constexpr size_t value = array_trait<T>::n;
};

template <typename T>
constexpr size_t tuple_size_v = tuple_size<T>::value;

//
//
//
// A wrapper for `boost::pfr::get`, which takes C arrays into account.
// For example, `int v[3] = {0, 1, 2}`, then
// `get<0>(v)`, `get<1>(v)`, and `get<2>(v)` will
// return references to `0`, `1`, and `2`.

// For non-array types, simply call `boost::pfr::get`.
template <auto iNonRec, typename T>
  requires(!std::is_array_v<std::remove_cvref_t<T>>)
[[nodiscard]] static constexpr decltype(auto) get(T &&v) noexcept {
  //! The type get by perfect forwarding.
  using TCheck0 = std::remove_cvref_t<decltype(boost::pfr::get<iNonRec>(std::forward<T>(v)))>;
  //! The type get by l-value reference.
  using TCheck1 = std::remove_cvref_t<decltype(boost::pfr::get<iNonRec>(std::declval<            //
                                                                        std::remove_cvref_t<T> & //
                                                                        >()))>;
  //! Check `boost::pfr` limitations and undefined behaviors.
  static_assert(std::is_same_v<TCheck0, TCheck1>,
                "`get` is currently not supported for this type because of the limitations of `boost::pfr`");

  return boost::pfr::get<iNonRec>(std::forward<T>(v));
}

// For array types.
template <auto iNonRec, typename T>
  requires(std::is_array_v<std::remove_cvref_t<T>>)
[[nodiscard]] static constexpr decltype(auto) get(T &&v) noexcept {
  // For l-value references, return references.
  if constexpr (std::is_lvalue_reference_v<std::remove_const_t<decltype(v)>>) {
    return std::forward<T>(v)[iNonRec];
  }
  // Else, return copies.
  else {
    auto res = Auto(std::forward<T>(v)[iNonRec]);
    return res;
  }
}

//
//
//
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
//
//! Unfortunately, `const`, `volatile`, and references are not allowed and
//! will be automatically considered as their `std::remove_cvref_t`-versions.
template <typename T_>
[[nodiscard]] static consteval bool IsMosaicPatternImpl() {
  using T = std::remove_cvref_t<T_>;

  //! For non-array types, the "strongest" copy and move ability is required.
  if (!std::is_array_v<T> && !(std::is_copy_constructible_v<T> && std::is_copy_assignable_v<T> &&
                               std::is_move_constructible_v<T> && std::is_move_assignable_v<T>))
    return false;

  // For aggregate types (structures, or array types).
  if constexpr (std::is_aggregate_v<T>) {
    //! `> 0` is required here because empty classes are considered unnecessary.
    if (tuple_size_v<T> == 0)
      return false;

    bool res = true;
    ForEach<tuple_size_v<T>>([&](auto i) {
      //! Here, `std::declval<T &>` is used instead of `std::declval<T>` in order to
      //! prevent array types such as `int[3]` from being converted to `int*`.
      using U = decltype(get<i>(std::declval<T &>()));
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
template <MosaicPattern T_>
[[nodiscard]] static consteval auto TupleSizeRecursiveImpl() {
  using T = std::remove_cvref_t<T_>;
  using TInteger = std::decay_t<decltype(tuple_size_v<T>)>;

  if constexpr (std::is_aggregate_v<T>) {
    TInteger sum = 0;
    ForEach<tuple_size_v<T>>([&](auto i) {
      using U = decltype(get<i>(std::declval<T &>()));
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
template <auto iRec, MosaicPattern T_>
[[nodiscard]] static consteval auto IRec2INonRec() {
  using T = std::remove_cvref_t<T_>;
  using TInteger = std::decay_t<decltype(tuple_size_v<T>)>;

  TInteger sum = 0;
  TInteger iNonRec = 0;

  ForEach<tuple_size_v<T>>([&](auto i) {
    using U = decltype(get<i>(std::declval<T &>()));
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
template <auto iNonRec, MosaicPattern T_>
[[nodiscard]] static consteval auto INonRec2IRec() {
  using T = std::remove_cvref_t<T_>;
  using TInteger = std::decay_t<decltype(tuple_size_v<T>)>;

  TInteger sum = 0;

  ForEach<tuple_size_v<T>>([&](auto i) {
    if (i >= iNonRec)
      return;

    using U = decltype(get<i>(std::declval<T &>()));
    sum += tuple_size_recursive_v<U>;
  });

  return sum;
}

//
//
//
// We should be able to recursively or non-recursively
// iterate through all the elements of a `MosaicPattern` instance.
template <auto iRec, MosaicPattern T>
[[nodiscard]] static inline constexpr decltype(auto) get_recursive(T &&v) noexcept {
  using TCleaned = std::remove_cvref_t<T>;
  static_assert(iRec < tuple_size_recursive_v<TCleaned>, "Index out of range");

  using TInteger = std::decay_t<decltype(tuple_size_v<TCleaned>)>;

  if constexpr (std::is_aggregate_v<TCleaned>) {
    constexpr TInteger iNonRec = IRec2INonRec<iRec, TCleaned>();
    return get_recursive<iRec - INonRec2IRec<iNonRec, TCleaned>()>(get<iNonRec>(std::forward<T>(v)));
  } else {
    return get<iRec>(std::forward<T>(v));
  }
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
      MakeTypeArray<TArray, std::remove_cvref_t<decltype(get_recursive<i>(std::declval<T &>()))>>>::type;
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

template <typename T_, typename TMosaicPattern_>
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

  static_assert(MosaicPattern<P>, "The given type should satisfy `MosaicPattern`");

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
